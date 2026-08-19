package com.argus.vision.ai

import android.graphics.Bitmap
import android.util.Base64
import com.argus.vision.settings.ArgusSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * OPTIONAL cloud reasoning fallback. Disabled by default; only runs when the user has
 * explicitly enabled it in Settings, provided credentials, and asked a question the
 * local QueryEngine could not answer. Sends ONE downscaled frame per explicit question -
 * never a continuous stream. Uses the OpenAI-compatible chat/completions schema so any
 * compatible endpoint works.
 *
 * The architecture also leaves room for a LOCAL VLM here: implement this same call with
 * MediaPipe LLM Inference (e.g. Gemma 3n, ~3 GB on-device) and nothing else changes.
 */
object CloudReasoner {

    suspend fun ask(
        settings: ArgusSettings,
        question: String,
        frame: Bitmap,
        detectionSummary: String,
    ): String = withContext(Dispatchers.IO) {
        val scaled = downscale(frame, 640)
        val jpeg = ByteArrayOutputStream().use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, 70, out)
            out.toByteArray()
        }
        val b64 = Base64.encodeToString(jpeg, Base64.NO_WRAP)

        val body = JSONObject().apply {
            put("model", settings.cloudModel)
            put("max_tokens", 200)
            put("messages", JSONArray().put(
                JSONObject().apply {
                    put("role", "user")
                    put("content", JSONArray()
                        .put(JSONObject().apply {
                            put("type", "text")
                            put(
                                "text",
                                "You are ARGUS, a concise heads-up-display vision assistant. " +
                                    "Local object detector reports: $detectionSummary. " +
                                    "Answer briefly: $question"
                            )
                        })
                        .put(JSONObject().apply {
                            put("type", "image_url")
                            put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$b64"))
                        })
                    )
                }
            ))
        }

        val conn = (URL(settings.cloudEndpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15000
            readTimeout = 60000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Authorization", "Bearer ${settings.cloudApiKey}")
        }
        try {
            conn.outputStream.use { it.write(body.toString().toByteArray()) }
            val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
            val text = stream.bufferedReader().use { it.readText() }
            if (conn.responseCode !in 200..299) {
                return@withContext "Cloud reasoning failed (${conn.responseCode})."
            }
            JSONObject(text)
                .getJSONArray("choices").getJSONObject(0)
                .getJSONObject("message").getString("content").trim()
        } catch (t: Throwable) {
            "Cloud reasoning unavailable: ${t.message ?: "network error"}"
        } finally {
            conn.disconnect()
        }
    }

    private fun downscale(src: Bitmap, maxDim: Int): Bitmap {
        val scale = maxDim.toFloat() / maxOf(src.width, src.height)
        if (scale >= 1f) return src
        return Bitmap.createScaledBitmap(
            src, (src.width * scale).toInt(), (src.height * scale).toInt(), true
        )
    }
}
