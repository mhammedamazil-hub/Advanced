package com.argus.vision

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.argus.vision.ai.CloudReasoner
import com.argus.vision.perf.PerformanceMonitor
import com.argus.vision.settings.ArgusSettings
import com.argus.vision.settings.DetectionMode
import com.argus.vision.settings.SettingsStore
import com.argus.vision.vision.VisionEngine
import com.argus.vision.vision.model.ModelManager
import com.argus.vision.voice.QueryEngine
import com.argus.vision.voice.VoiceInterface
import com.argus.vision.voice.VoiceState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

data class AssistantMessage(val query: String, val answer: String)

data class SystemSample(val cpuPercent: Float = 0f, val usedMemMb: Int = 0, val totalMemMb: Int = 0)

@OptIn(FlowPreview::class)
class MainViewModel(app: Application) : AndroidViewModel(app) {

    val settingsStore = SettingsStore(app)
    val engine = VisionEngine(app)
    private val perfMonitor = PerformanceMonitor(app)

    private val _voiceState = MutableStateFlow<VoiceState>(VoiceState.Idle)
    val voiceState: StateFlow<VoiceState> = _voiceState.asStateFlow()

    private val _assistant = MutableStateFlow<AssistantMessage?>(null)
    val assistant: StateFlow<AssistantMessage?> = _assistant.asStateFlow()

    private val _system = MutableStateFlow(SystemSample())
    val system: StateFlow<SystemSample> = _system.asStateFlow()

    /** Track id to visually highlight (e.g. after "find my phone"). */
    private val _focusTrackId = MutableStateFlow<Int?>(null)
    val focusTrackId: StateFlow<Int?> = _focusTrackId.asStateFlow()

    private val voice = VoiceInterface(
        context = app,
        onStateChange = { state ->
            _voiceState.value = state
            if (state is VoiceState.Error) scheduleVoiceReset()
        },
        onQuery = ::handleQuery,
    )

    private var dismissJob: Job? = null
    private var voiceResetJob: Job? = null

    init {
        // Load the model matching the persisted mode, off the main thread.
        // Debounced so dragging the threshold slider doesn't reload the model per tick.
        viewModelScope.launch(Dispatchers.Default) {
            settingsStore.settings
                .map { it.mode to it.scoreThreshold }
                .distinctUntilChanged()
                .debounce(350)
                .collect { (mode, threshold) -> applyMode(mode, threshold) }
        }
        // Sample system metrics only while the perf panel is visible.
        viewModelScope.launch(Dispatchers.Default) {
            while (true) {
                if (settingsStore.settings.value.showPerfPanel) {
                    val (used, total) = perfMonitor.sampleMemoryMb()
                    _system.value = SystemSample(perfMonitor.sampleCpuPercent(), used, total)
                }
                delay(1000)
            }
        }
    }

    private fun applyMode(mode: DetectionMode, threshold: Float) {
        val mm = engine.modelManager
        val spec = when (mode) {
            DetectionMode.FAST -> mm.byId(ModelManager.FAST_ID)
            DetectionMode.BALANCED -> mm.byId(ModelManager.FAST_ID)
            DetectionMode.ACCURATE -> mm.byId(ModelManager.ACCURATE_ID)
        } ?: mm.defaultModel()

        engine.modeMinIntervalMs = when (mode) {
            DetectionMode.FAST -> 0L        // uncapped - paced by inference speed only
            DetectionMode.BALANCED -> 70L   // ~14 Hz inference, tracker keeps HUD smooth
            DetectionMode.ACCURATE -> 0L    // lite2 is naturally slower
        }

        if (spec != null) {
            engine.loadModel(spec, threshold)
        }
    }

    // ---------------- Voice ----------------

    fun onMicPressed() {
        voiceResetJob?.cancel()
        if (_voiceState.value is VoiceState.Listening) {
            voice.stopListening()
            _voiceState.value = VoiceState.Idle
        } else {
            voice.startListening()
        }
    }

    private fun handleQuery(query: String) {
        viewModelScope.launch(Dispatchers.Default) {
            val (w, h) = engine.frameSize.value
            val tracks = engine.tracks.value
            val local = QueryEngine.answer(query, tracks, engine.latestBitmap, w, h)
            _focusTrackId.value = local.focusTrackId

            val settings = settingsStore.settings.value
            val answer = if (local.answered) {
                local.text
            } else if (settings.cloudEnabled && settings.cloudApiKey.isNotBlank()) {
                val frame = engine.latestBitmap
                if (frame != null) {
                    postMessage(query, "Consulting reasoning model...", speak = false)
                    CloudReasoner.ask(settings, query, frame, QueryEngine.summarize(tracks))
                } else local.text
            } else {
                local.text + " Enable cloud reasoning in settings for deeper answers."
            }
            postMessage(query, answer, speak = settings.speakAnswers)
            scheduleVoiceReset()
        }
    }

    private fun postMessage(query: String, answer: String, speak: Boolean) {
        _assistant.value = AssistantMessage(query, answer)
        if (speak) voice.speak(answer)
        dismissJob?.cancel()
        dismissJob = viewModelScope.launch {
            delay(10_000)
            _assistant.value = null
            _focusTrackId.value = null
        }
    }

    private fun scheduleVoiceReset() {
        voiceResetJob?.cancel()
        voiceResetJob = viewModelScope.launch {
            delay(2500)
            _voiceState.value = VoiceState.Idle
        }
    }

    override fun onCleared() {
        voice.shutdown()
        engine.close()
    }
}
