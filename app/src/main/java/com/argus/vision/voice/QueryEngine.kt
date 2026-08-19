package com.argus.vision.voice

import android.graphics.Bitmap
import android.graphics.Color
import com.argus.vision.vision.tracker.TrackedObject
import java.util.Locale

/**
 * Answers questions locally from the detector's structured results - no LLM, no network.
 * Handles: "what do you see", "count the bottles", "find my phone", "what is that",
 * "what color is the ...", and label containment lookups.
 */
object QueryEngine {

    /** Result of local parsing. If [answered] is false, a reasoning model would be needed. */
    data class Answer(val text: String, val answered: Boolean, val focusTrackId: Int? = null)

    fun answer(
        rawQuery: String,
        tracks: List<TrackedObject>,
        frame: Bitmap?,
        frameWidth: Int,
        frameHeight: Int,
    ): Answer {
        val q = rawQuery.lowercase(Locale.US).trim()

        // "what do you see" / "what objects are around me" / "what's around"
        if (Regex("what (do you|can you) see|what objects|what('s| is) around|scan|describe the scene").containsMatchIn(q)) {
            return Answer(summarize(tracks), true)
        }

        // "count the bottles" / "how many people"
        Regex("(?:count(?: the)?|how many) ([a-z ]+?)s?(?: are.*)?$").find(q)?.let { m ->
            val target = m.groupValues[1].trim()
            val label = matchLabel(target, tracks) ?: canonical(target)
            val n = tracks.count { it.label.lowercase(Locale.US).contains(label) }
            return Answer(
                if (n == 0) "I don't see any ${plural(label)} right now."
                else "I can see $n ${if (n == 1) label else plural(label)}.",
                true
            )
        }

        // "find my phone" / "where is the cup"
        Regex("(?:find|where(?:'s| is)|locate)(?: my| the| a)? ([a-z ]+)").find(q)?.let { m ->
            val target = canonical(m.groupValues[1].trim())
            val hit = tracks
                .filter { it.label.lowercase(Locale.US).contains(target) }
                .maxByOrNull { it.confidence }
                ?: return Answer("No $target in view. Keep scanning and I'll keep looking.", true)
            return Answer(
                "${hit.label.lowercase(Locale.US).replaceFirstChar { it.uppercase() }} detected, " +
                    "${position(hit, frameWidth, frameHeight)}, " +
                    "confidence ${"%.0f".format(hit.confidence * 100)} percent.",
                true,
                focusTrackId = hit.id,
            )
        }

        // "what color is the car"
        Regex("what colou?r is (?:the |that |this |my )?([a-z ]+)").find(q)?.let { m ->
            val target = canonical(m.groupValues[1].trim())
            val hit = tracks
                .filter { it.label.lowercase(Locale.US).contains(target) }
                .maxByOrNull { it.box.width() * it.box.height() }
                ?: return Answer("I don't see a $target right now.", true)
            val color = frame?.let { dominantColorName(it, hit) }
                ?: return Answer("No frame available to inspect.", true)
            return Answer("The $target appears to be mostly $color.", true, focusTrackId = hit.id)
        }

        // "what is that / this"
        if (Regex("what (is|am i looking at)( th(at|is))?|identify").containsMatchIn(q)) {
            val cx = frameWidth / 2f
            val cy = frameHeight / 2f
            val hit = tracks.minByOrNull {
                val dx = it.box.centerX() - cx
                val dy = it.box.centerY() - cy
                dx * dx + dy * dy
            } ?: return Answer("Nothing recognized in view yet.", true)
            return Answer(
                "That looks like a ${hit.label.lowercase(Locale.US)}, " +
                    "confidence ${"%.0f".format(hit.confidence * 100)} percent.",
                true,
                focusTrackId = hit.id,
            )
        }

        // Direct label mention: "phone?" "is there a person"
        matchLabel(q, tracks)?.let { label ->
            val n = tracks.count { it.label.lowercase(Locale.US).contains(label) }
            return Answer("Yes - $n ${if (n == 1) label else plural(label)} in view.", true)
        }

        return Answer(
            "I can't answer that from local detection alone.",
            answered = false,
        )
    }

    fun summarize(tracks: List<TrackedObject>): String {
        if (tracks.isEmpty()) return "No objects recognized in the current view."
        val counts = tracks.groupingBy { it.label.lowercase(Locale.US) }.eachCount()
        val parts = counts.entries
            .sortedByDescending { it.value }
            .joinToString(", ") { (label, n) -> if (n == 1) "one $label" else "$n ${plural(label)}" }
        return "I can see $parts."
    }

    private fun matchLabel(text: String, tracks: List<TrackedObject>): String? =
        tracks.map { it.label.lowercase(Locale.US) }.distinct()
            .firstOrNull { text.contains(it) || text.contains(plural(it)) }

    private fun canonical(word: String): String {
        val w = word.removeSuffix("s").trim()
        return mapOf(
            "phone" to "cell phone", "mobile" to "cell phone", "smartphone" to "cell phone",
            "computer" to "laptop", "table" to "dining table", "television" to "tv",
            "bike" to "bicycle", "bag" to "backpack", "sofa" to "couch",
        )[w] ?: w
    }

    private fun plural(label: String): String =
        if (label == "person") "people"
        else if (label.endsWith("s")) label else label + "s"

    private fun position(t: TrackedObject, w: Int, h: Int): String {
        val x = t.box.centerX() / w
        val y = t.box.centerY() / h
        val horiz = when {
            x < 0.36f -> "on the left"
            x > 0.64f -> "on the right"
            else -> "in the center"
        }
        val vert = when {
            y < 0.33f -> ", upper part of the view"
            y > 0.67f -> ", lower part of the view"
            else -> ""
        }
        return horiz + vert
    }

    /** Cheap dominant-color estimate over the object's box (sampled, not per-pixel). */
    private fun dominantColorName(bmp: Bitmap, t: TrackedObject): String {
        val l = t.box.left.toInt().coerceIn(0, bmp.width - 1)
        val tp = t.box.top.toInt().coerceIn(0, bmp.height - 1)
        val r = t.box.right.toInt().coerceIn(l + 1, bmp.width)
        val b = t.box.bottom.toInt().coerceIn(tp + 1, bmp.height)
        val stepX = ((r - l) / 24).coerceAtLeast(1)
        val stepY = ((b - tp) / 24).coerceAtLeast(1)
        var sr = 0L; var sg = 0L; var sb = 0L; var n = 0
        var y = tp + (b - tp) / 6           // bias toward the object center
        val yEnd = b - (b - tp) / 6
        while (y < yEnd) {
            var x = l + (r - l) / 6
            val xEnd = r - (r - l) / 6
            while (x < xEnd) {
                val c = bmp.getPixel(x, y)
                sr += Color.red(c); sg += Color.green(c); sb += Color.blue(c); n++
                x += stepX
            }
            y += stepY
        }
        if (n == 0) return "unclear"
        val hsv = FloatArray(3)
        Color.RGBToHSV((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt(), hsv)
        val (hue, sat, v) = hsv
        return when {
            v < 0.18f -> "black"
            sat < 0.14f && v > 0.82f -> "white"
            sat < 0.16f -> "gray"
            hue < 15f || hue >= 345f -> "red"
            hue < 45f -> "orange"
            hue < 70f -> "yellow"
            hue < 165f -> "green"
            hue < 200f -> "cyan"
            hue < 255f -> "blue"
            hue < 290f -> "purple"
            hue < 345f -> "pink"
            else -> "unclear"
        }
    }
}
