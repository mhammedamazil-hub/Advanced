package com.argus.vision.vision.model

import android.content.Context
import java.io.File

/**
 * Describes a detection model the engine can load.
 * The app is deliberately NOT hard-coded around one model: built-in models ship in
 * assets/models/ and any .tflite file (TFLite object-detection model with metadata)
 * dropped into <app-files>/models/ appears here automatically as a custom model.
 */
data class ModelSpec(
    val id: String,
    val displayName: String,
    /** Path inside assets/ for built-in models, or absolute file path for custom ones. */
    val path: String,
    val isAsset: Boolean,
    val inputSize: Int,
    val description: String,
)

class ModelManager(private val context: Context) {

    companion object {
        const val FAST_ID = "efficientdet_lite0"
        const val ACCURATE_ID = "efficientdet_lite2"
    }

    private val builtIn = listOf(
        ModelSpec(
            id = FAST_ID,
            displayName = "EfficientDet-Lite0",
            path = "models/efficientdet_lite0.tflite",
            isAsset = true,
            inputSize = 320,
            description = "Fast detection model - lowest latency, 80 COCO classes",
        ),
        ModelSpec(
            id = ACCURATE_ID,
            displayName = "EfficientDet-Lite2",
            path = "models/efficientdet_lite2.tflite",
            isAsset = true,
            inputSize = 448,
            description = "Accurate detection model - higher latency, 80 COCO classes",
        ),
    )

    /** Directory users (or a future in-app downloader) can drop custom .tflite models into. */
    val customModelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun availableModels(): List<ModelSpec> {
        val assetNames = runCatching {
            context.assets.list("models")?.toSet() ?: emptySet()
        }.getOrDefault(emptySet())

        val present = builtIn.filter { it.path.substringAfterLast('/') in assetNames }

        val custom = customModelDir.listFiles { f -> f.extension == "tflite" }
            ?.map { f ->
                ModelSpec(
                    id = "custom_${f.nameWithoutExtension}",
                    displayName = "Custom: ${f.nameWithoutExtension}",
                    path = f.absolutePath,
                    isAsset = false,
                    inputSize = 320,
                    description = "User-provided model",
                )
            } ?: emptyList()

        return present + custom
    }

    fun byId(id: String): ModelSpec? = availableModels().firstOrNull { it.id == id }

    fun defaultModel(): ModelSpec? =
        byId(FAST_ID) ?: availableModels().firstOrNull()
}
