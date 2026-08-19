package com.argus.vision.vision.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.argus.vision.vision.model.ModelSpec
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import java.io.File
import java.nio.ByteBuffer

/**
 * Detector abstraction so the runtime can be swapped later
 * (e.g. an ONNX Runtime or custom LiteRT implementation).
 */
interface FrameDetector {
    val modelName: String
    val usingGpu: Boolean
    /** Submit a frame. Returns false if the detector is busy and the frame was skipped. */
    fun detectAsync(bitmap: Bitmap, timestampMs: Long): Boolean
    fun close()
}

/**
 * MediaPipe Tasks (LiteRT/TFLite) implementation.
 *
 * Runs in LIVE_STREAM mode: detectAsync() returns immediately and results arrive on a
 * background thread, so the camera and UI threads are never blocked. Tries the GPU
 * delegate first (Xclipse 940 on the Galaxy S24), falls back to CPU/XNNPACK.
 */
class MediaPipeDetector private constructor(
    private val detector: ObjectDetector,
    override val modelName: String,
    override val usingGpu: Boolean,
) : FrameDetector {

    @Volatile private var busy = false

    override fun detectAsync(bitmap: Bitmap, timestampMs: Long): Boolean {
        if (busy) return false
        busy = true
        return try {
            detector.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
            true
        } catch (t: Throwable) {
            Log.e(TAG, "detectAsync failed", t)
            busy = false
            false
        }
    }

    internal fun onResultDelivered() {
        busy = false
    }

    override fun close() = runCatching { detector.close() }.let { }

    companion object {
        private const val TAG = "MediaPipeDetector"

        /**
         * Create a detector. Attempts GPU delegate first, CPU second.
         * Throws if both fail (caller shows the error state and keeps the camera alive).
         */
        fun create(
            context: Context,
            spec: ModelSpec,
            scoreThreshold: Float,
            maxResults: Int,
            onResult: (DetectionFrame) -> Unit,
            onError: (Throwable) -> Unit,
        ): MediaPipeDetector {
            var lastError: Throwable? = null
            for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
                var self: MediaPipeDetector? = null
                try {
                    val baseBuilder = BaseOptions.builder().setDelegate(delegate)
                    if (spec.isAsset) {
                        baseBuilder.setModelAssetPath(spec.path)
                    } else {
                        val bytes = File(spec.path).readBytes()
                        val buffer = ByteBuffer.allocateDirect(bytes.size).apply {
                            put(bytes); rewind()
                        }
                        baseBuilder.setModelAssetBuffer(buffer)
                    }

                    val options = ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(baseBuilder.build())
                        .setRunningMode(RunningMode.LIVE_STREAM)
                        .setScoreThreshold(scoreThreshold)
                        .setMaxResults(maxResults)
                        .setResultListener { result: ObjectDetectorResult, input ->
                            self?.onResultDelivered()
                            onResult(result.toFrame(input.width, input.height))
                        }
                        .setErrorListener { e ->
                            self?.onResultDelivered()
                            onError(e)
                        }
                        .build()

                    val det = ObjectDetector.createFromOptions(context, options)
                    self = MediaPipeDetector(det, spec.displayName, delegate == Delegate.GPU)
                    Log.i(TAG, "Loaded ${spec.displayName} with $delegate delegate")
                    return self
                } catch (t: Throwable) {
                    Log.w(TAG, "Init with $delegate failed, trying next", t)
                    lastError = t
                }
            }
            throw IllegalStateException("Could not initialize detector", lastError)
        }

        private fun ObjectDetectorResult.toFrame(w: Int, h: Int): DetectionFrame {
            val list = detections().mapNotNull { d ->
                val cat = d.categories().maxByOrNull { it.score() } ?: return@mapNotNull null
                Detection(
                    label = cat.categoryName().uppercase(),
                    confidence = cat.score(),
                    box = d.boundingBox(),
                )
            }
            return DetectionFrame(
                detections = list,
                frameWidth = w,
                frameHeight = h,
                timestampMs = timestampMs(),
                inferenceTimeMs = 0L, // filled in by the engine (submit -> result wall time)
            )
        }
    }
}
