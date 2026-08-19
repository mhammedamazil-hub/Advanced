package com.argus.vision.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.argus.vision.perf.PerfStats
import com.argus.vision.vision.detector.DetectionFrame
import com.argus.vision.vision.detector.FrameDetector
import com.argus.vision.vision.detector.MediaPipeDetector
import com.argus.vision.vision.model.ModelManager
import com.argus.vision.vision.model.ModelSpec
import com.argus.vision.vision.tracker.IouTracker
import com.argus.vision.vision.tracker.TrackedObject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicLong

sealed interface EngineState {
    data object Initializing : EngineState
    /** Vision online. */
    data class Online(val modelName: String, val gpu: Boolean) : EngineState
    /** Model failed to load - camera keeps running, HUD shows the error. */
    data class Error(val message: String) : EngineState
}

/**
 * VisionEngine - owns the detector + tracker and adapts performance at runtime.
 *
 * Threading model:
 *  - onFrame() is called on the CameraX analyzer thread (never the UI thread).
 *  - MediaPipe LIVE_STREAM results arrive on MediaPipe's own background thread.
 *  - The UI only observes StateFlows; nothing here ever blocks the main thread.
 *
 * Pacing model ("frame skipping / throttling"):
 *  - A frame is only submitted when the detector is idle, so inference naturally runs
 *    as fast as the hardware allows without ever building a queue (bounded memory).
 *  - A thermal-aware minimum interval is added on top: when Android reports the device
 *    is heating up, inference frequency is reduced instead of freezing the camera.
 */
class VisionEngine(
    private val context: Context,
    val modelManager: ModelManager = ModelManager(context),
) {
    private val _state = MutableStateFlow<EngineState>(EngineState.Initializing)
    val state: StateFlow<EngineState> = _state.asStateFlow()

    private val _tracks = MutableStateFlow<List<TrackedObject>>(emptyList())
    val tracks: StateFlow<List<TrackedObject>> = _tracks.asStateFlow()

    /** Rotated analysis-frame size, needed by the HUD for coordinate mapping. */
    private val _frameSize = MutableStateFlow(Pair(480, 640))
    val frameSize: StateFlow<Pair<Int, Int>> = _frameSize.asStateFlow()

    private val _perf = MutableStateFlow(PerfStats())
    val perf: StateFlow<PerfStats> = _perf.asStateFlow()

    private val tracker = IouTracker()
    @Volatile private var detector: FrameDetector? = null
    @Volatile private var currentSpec: ModelSpec? = null

    /** Latest upright RGB frame, kept for local color queries / optional cloud reasoning. */
    @Volatile var latestBitmap: Bitmap? = null
        private set

    private val lastSubmitUptime = AtomicLong(0)
    private val submitWallTime = AtomicLong(0)

    // Rolling metrics
    @Volatile private var emaLatencyMs = 0.0
    @Volatile private var cameraFrames = 0
    @Volatile private var inferenceFrames = 0
    @Volatile private var windowStart = SystemClock.uptimeMillis()

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    /**
     * Base pacing set by the detection mode (FAST = uncapped, BALANCED caps inference
     * around ~14 Hz to save battery - visual smoothness comes from the tracker).
     */
    @Volatile var modeMinIntervalMs: Long = 70L

    private fun effectiveMinIntervalMs(): Long =
        maxOf(modeMinIntervalMs, thermalMinIntervalMs())

    /** Extra pacing interval driven by thermal state (adaptive performance mode). */
    private fun thermalMinIntervalMs(): Long = when (powerManager.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> 0L
        PowerManager.THERMAL_STATUS_MODERATE -> 120L
        PowerManager.THERMAL_STATUS_SEVERE -> 300L
        else -> 600L // CRITICAL and above: keep camera fluid, detect ~1.6x/sec
    }

    fun thermalStatusLabel(): String = when (powerManager.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "NOMINAL"
        PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
        PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
        else -> "EMERGENCY"
    }

    /** (Re)load a model. Safe to call while running; called off the main thread. */
    @Synchronized
    fun loadModel(spec: ModelSpec, scoreThreshold: Float) {
        _state.value = EngineState.Initializing
        detector?.close()
        detector = null
        tracker.clear()
        _tracks.value = emptyList()
        try {
            val det = MediaPipeDetector.create(
                context = context,
                spec = spec,
                scoreThreshold = scoreThreshold,
                maxResults = 12,
                onResult = ::onDetectionResult,
                onError = { e -> Log.e(TAG, "Inference error", e) },
            )
            detector = det
            currentSpec = spec
            _state.value = EngineState.Online(det.modelName, det.usingGpu)
        } catch (t: Throwable) {
            Log.e(TAG, "Model init failed", t)
            _state.value = EngineState.Error(
                "VISION MODEL FAILED TO INITIALIZE\n${t.message ?: "unknown error"}"
            )
        }
    }

    /** Called by CameraX for every analysis frame. MUST close the proxy. */
    fun onFrame(image: ImageProxy) {
        try {
            cameraFrames++
            maybePublishStats()

            val det = detector ?: return
            val now = SystemClock.uptimeMillis()
            if (now - lastSubmitUptime.get() < effectiveMinIntervalMs()) return

            // Convert + rotate to upright orientation. Only done for frames we actually
            // submit (skipped frames cost nothing but the YUV close).
            val rotation = image.imageInfo.rotationDegrees
            var bmp = image.toBitmap()
            if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
            }

            if (det.detectAsync(bmp, now)) {
                latestBitmap = bmp
                lastSubmitUptime.set(now)
                submitWallTime.set(now)
                _frameSize.value = Pair(bmp.width, bmp.height)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "onFrame failed", t)
        } finally {
            image.close()
        }
    }

    private fun onDetectionResult(frame: DetectionFrame) {
        val latency = SystemClock.uptimeMillis() - submitWallTime.get()
        emaLatencyMs = if (emaLatencyMs == 0.0) latency.toDouble()
        else emaLatencyMs * 0.85 + latency * 0.15
        inferenceFrames++
        _tracks.value = tracker.update(frame.detections)
        maybePublishStats()
    }

    private fun maybePublishStats() {
        val now = SystemClock.uptimeMillis()
        val elapsed = now - windowStart
        if (elapsed < 500) return
        synchronized(this) {
            val e = now - windowStart
            if (e < 500) return
            _perf.value = _perf.value.copy(
                cameraFps = cameraFrames * 1000f / e,
                inferenceFps = inferenceFrames * 1000f / e,
                inferenceLatencyMs = emaLatencyMs.toFloat(),
                modelName = currentSpec?.displayName ?: "-",
                delegate = if (detector?.usingGpu == true) "GPU" else "CPU",
                objectCount = _tracks.value.size,
                thermal = thermalStatusLabel(),
                frameWidth = _frameSize.value.first,
                frameHeight = _frameSize.value.second,
            )
            cameraFrames = 0
            inferenceFrames = 0
            windowStart = now
        }
    }

    fun close() {
        detector?.close()
        detector = null
    }

    private companion object {
        const val TAG = "VisionEngine"
    }
}
