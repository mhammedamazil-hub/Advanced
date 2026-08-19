package com.argus.vision.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

enum class DetectionMode { FAST, BALANCED, ACCURATE }

data class ArgusSettings(
    val mode: DetectionMode = DetectionMode.BALANCED,
    val scoreThreshold: Float = 0.40f,
    val showPerfPanel: Boolean = false,
    val showTrackIds: Boolean = true,
    val speakAnswers: Boolean = true,
    val scanlineEnabled: Boolean = true,
    // Optional cloud reasoning - OFF by default. Detection never uses the network.
    val cloudEnabled: Boolean = false,
    val cloudEndpoint: String = "https://api.openai.com/v1/chat/completions",
    val cloudApiKey: String = "",
    val cloudModel: String = "gpt-4o-mini",
)

/** Tiny SharedPreferences-backed settings store exposed as a StateFlow. */
class SettingsStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences("argus_settings", Context.MODE_PRIVATE)

    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<ArgusSettings> = _settings

    private fun load() = ArgusSettings(
        mode = runCatching {
            DetectionMode.valueOf(prefs.getString("mode", "BALANCED")!!)
        }.getOrDefault(DetectionMode.BALANCED),
        scoreThreshold = prefs.getFloat("scoreThreshold", 0.40f),
        showPerfPanel = prefs.getBoolean("showPerfPanel", false),
        showTrackIds = prefs.getBoolean("showTrackIds", true),
        speakAnswers = prefs.getBoolean("speakAnswers", true),
        scanlineEnabled = prefs.getBoolean("scanlineEnabled", true),
        cloudEnabled = prefs.getBoolean("cloudEnabled", false),
        cloudEndpoint = prefs.getString("cloudEndpoint", ArgusSettings().cloudEndpoint)!!,
        cloudApiKey = prefs.getString("cloudApiKey", "")!!,
        cloudModel = prefs.getString("cloudModel", ArgusSettings().cloudModel)!!,
    )

    fun update(transform: (ArgusSettings) -> ArgusSettings) {
        val s = transform(_settings.value)
        _settings.value = s
        prefs.edit()
            .putString("mode", s.mode.name)
            .putFloat("scoreThreshold", s.scoreThreshold)
            .putBoolean("showPerfPanel", s.showPerfPanel)
            .putBoolean("showTrackIds", s.showTrackIds)
            .putBoolean("speakAnswers", s.speakAnswers)
            .putBoolean("scanlineEnabled", s.scanlineEnabled)
            .putBoolean("cloudEnabled", s.cloudEnabled)
            .putString("cloudEndpoint", s.cloudEndpoint)
            .putString("cloudApiKey", s.cloudApiKey)
            .putString("cloudModel", s.cloudModel)
            .apply()
    }
}
