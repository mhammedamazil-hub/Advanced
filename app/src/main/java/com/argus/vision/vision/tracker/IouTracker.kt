package com.argus.vision.vision.tracker

import android.graphics.RectF
import com.argus.vision.vision.detector.Detection

/** A detection with a stable identity across frames. */
data class TrackedObject(
    val id: Int,
    val label: String,
    val confidence: Float,
    /** Smoothed box in rotated-frame pixel coordinates. */
    val box: RectF,
    /** How many consecutive inference cycles this object has been seen. */
    val hits: Int,
    /** Consecutive cycles it has been missing (kept alive briefly for stability). */
    val misses: Int,
)

/**
 * Lightweight IoU-based tracker.
 *
 * Greedy IoU matching + exponential smoothing of box coordinates. This keeps labels and
 * IDs stable while the camera moves, hides single-frame detector flicker, and means the
 * HUD does not depend on running inference at display refresh rate.
 * Zero heavy allocation: everything is small data classes at inference cadence (~10-40 Hz).
 */
class IouTracker(
    private val iouThreshold: Float = 0.35f,
    private val maxMisses: Int = 4,
    private val smoothing: Float = 0.45f, // weight of the NEW box
    private val minHitsToShow: Int = 2,
) {
    private var nextId = 1
    private val tracks = mutableListOf<TrackedObject>()

    @Synchronized
    fun update(detections: List<Detection>): List<TrackedObject> {
        val unmatchedDets = detections.toMutableList()
        val updated = mutableListOf<TrackedObject>()

        // Greedy match: best IoU pair first, same-label matches strongly preferred.
        for (track in tracks.sortedByDescending { it.hits }) {
            var best: Detection? = null
            var bestScore = 0f
            for (det in unmatchedDets) {
                val iou = iou(track.box, det.box)
                if (iou < iouThreshold) continue
                val score = if (det.label == track.label) iou + 1f else iou
                if (score > bestScore) {
                    bestScore = score
                    best = det
                }
            }
            if (best != null) {
                unmatchedDets.remove(best)
                updated += track.copy(
                    label = best.label,
                    confidence = best.confidence,
                    box = lerp(track.box, best.box, smoothing),
                    hits = track.hits + 1,
                    misses = 0,
                )
            } else if (track.misses < maxMisses) {
                // Keep briefly with decaying confidence so boxes don't blink.
                updated += track.copy(misses = track.misses + 1, confidence = track.confidence * 0.9f)
            }
        }

        for (det in unmatchedDets) {
            updated += TrackedObject(
                id = nextId++,
                label = det.label,
                confidence = det.confidence,
                box = RectF(det.box),
                hits = 1,
                misses = 0,
            )
        }

        tracks.clear()
        tracks.addAll(updated)
        return visible()
    }

    @Synchronized
    fun visible(): List<TrackedObject> =
        tracks.filter { it.hits >= minHitsToShow && it.misses <= 1 }

    @Synchronized
    fun clear() {
        tracks.clear()
    }

    private fun lerp(old: RectF, new: RectF, t: Float) = RectF(
        old.left + (new.left - old.left) * t,
        old.top + (new.top - old.top) * t,
        old.right + (new.right - old.right) * t,
        old.bottom + (new.bottom - old.bottom) * t,
    )

    private fun iou(a: RectF, b: RectF): Float {
        val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
        val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val inter = ix * iy
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }
}
