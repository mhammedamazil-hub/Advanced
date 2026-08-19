package com.argus.vision.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import java.util.Locale

sealed interface VoiceState {
    data object Idle : VoiceState
    data object Listening : VoiceState
    data class Heard(val text: String) : VoiceState
    data class Error(val message: String) : VoiceState
}

/**
 * On-device voice I/O. Uses Android's SpeechRecognizer with EXTRA_PREFER_OFFLINE
 * (the Galaxy S24 ships on-device recognition) and TextToSpeech for spoken answers.
 * The microphone is only active between startListening() and the result callback -
 * never in the background.
 */
class VoiceInterface(
    private val context: Context,
    private val onStateChange: (VoiceState) -> Unit,
    private val onQuery: (String) -> Unit,
) {
    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    fun isAvailable(): Boolean = SpeechRecognizer.isRecognitionAvailable(context)

    /** Must be called on the main thread. */
    fun startListening() {
        if (!isAvailable()) {
            onStateChange(VoiceState.Error("Speech recognition unavailable on this device"))
            return
        }
        stopListening()
        val r = SpeechRecognizer.createSpeechRecognizer(context)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) = onStateChange(VoiceState.Listening)
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}

            override fun onError(error: Int) {
                val msg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH,
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Didn't catch that"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission required"
                    else -> "Speech error ($error)"
                }
                onStateChange(VoiceState.Error(msg))
                release()
            }

            override fun onResults(results: Bundle?) {
                val text = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                release()
                if (text.isNullOrBlank()) {
                    onStateChange(VoiceState.Error("Didn't catch that"))
                } else {
                    onStateChange(VoiceState.Heard(text))
                    onQuery(text)
                }
            }
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        r.startListening(intent)
    }

    fun stopListening() = release()

    private fun release() {
        recognizer?.destroy()
        recognizer = null
    }

    fun speak(text: String) {
        if (tts == null) {
            tts = TextToSpeech(context) { status ->
                ttsReady = status == TextToSpeech.SUCCESS
                if (ttsReady) {
                    tts?.language = Locale.getDefault()
                    tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "argus")
                }
            }
        } else if (ttsReady) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "argus")
        }
    }

    fun shutdown() {
        release()
        tts?.shutdown()
        tts = null
    }
}
