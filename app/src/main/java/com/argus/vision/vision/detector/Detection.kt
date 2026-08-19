package com.argus.vision.vision.detector

import android.graphics.RectF

/** One raw detection coming out of a detector, in rotated-frame pixel coordinates. */
data class Detection(
    val label: String,
    val confidence: Float,
    val box: RectF,
)

/** A batch of detections for one frame. */
data class DetectionFrame(
    val detections: List<Detection>,
    val frameWidth: Int,
    val frameHeight: Int,
    val timestampMs: Long,
    val inferenceTimeMs: Long,
)
