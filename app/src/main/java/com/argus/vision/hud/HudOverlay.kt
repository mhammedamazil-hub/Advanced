package com.argus.vision.hud

import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import com.argus.vision.ui.theme.Hud
import com.argus.vision.vision.tracker.TrackedObject
import java.util.Locale
import kotlin.math.exp
import kotlin.math.min

/**
 * Futuristic HUD renderer.
 *
 * One Compose Canvas, one draw pass, GPU-composited. Text uses cached native Paints
 * (cheaper than Compose TextMeasurer at 60 Hz). Between inference results, boxes are
 * interpolated toward their targets every display frame, so the HUD glides at the
 * display refresh rate even when inference runs at 10-30 Hz.
 */
@Composable
fun HudOverlay(
    tracks: List<TrackedObject>,
    frameWidth: Int,
    frameHeight: Int,
    focusTrackId: Int?,
    showTrackIds: Boolean,
    scanlineEnabled: Boolean,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    val paints = remember(density) { HudPaints(density.density) }
    val displayed = remember { HashMap<Int, RectF>() }
    val timeHolder = remember { longArrayOf(0L) }

    // Single animation clock driving scanline sweep + reticle rotation + pulses.
    // Reading it inside the draw phase invalidates the canvas every display frame.
    val phase by rememberInfiniteTransition(label = "hud")
        .animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(tween(4000, easing = LinearEasing), RepeatMode.Restart),
            label = "phase",
        )

    Canvas(modifier = modifier) {
        val now = System.nanoTime()
        val dt = if (timeHolder[0] == 0L) 0.016f
        else ((now - timeHolder[0]) / 1e9f).coerceIn(0.001f, 0.1f)
        timeHolder[0] = now

        // --- coordinate mapping: analysis frame -> screen (cover / center-crop) ---
        val scale = if (frameWidth > 0 && frameHeight > 0)
            maxOf(size.width / frameWidth, size.height / frameHeight) else 1f
        val dx = (size.width - frameWidth * scale) / 2f
        val dy = (size.height - frameHeight * scale) / 2f

        // --- ambient layers (cheap: a few lines + one gradient band) ---
        drawScreenFrame(paints)
        if (scanlineEnabled) drawScanline(phase)
        drawReticle(phase, tracks.isEmpty())

        // --- tracked objects ---
        val lerpK = 1f - exp(-14f * dt) // frame-rate independent smoothing
        val liveIds = HashSet<Int>(tracks.size)
        for (t in tracks) {
            liveIds.add(t.id)
            val target = RectF(
                t.box.left * scale + dx,
                t.box.top * scale + dy,
                t.box.right * scale + dx,
                t.box.bottom * scale + dy,
            )
            val shown = displayed.getOrPut(t.id) { RectF(target) }
            shown.left += (target.left - shown.left) * lerpK
            shown.top += (target.top - shown.top) * lerpK
            shown.right += (target.right - shown.right) * lerpK
            shown.bottom += (target.bottom - shown.bottom) * lerpK

            drawDetection(shown, t, t.id == focusTrackId, showTrackIds, phase, paints)
        }
        displayed.keys.retainAll(liveIds)
    }
}

private fun DrawScope.drawDetection(
    r: RectF,
    t: TrackedObject,
    focused: Boolean,
    showId: Boolean,
    phase: Float,
    p: HudPaints,
) {
    val color = if (focused) Hud.Warning else Hud.Primary
    val dimColor = color.copy(alpha = if (focused) 0.55f else 0.30f)
    val w = r.width()
    val h = r.height()
    if (w < 4f || h < 4f) return

    // thin main box
    drawRect(
        color = dimColor,
        topLeft = Offset(r.left, r.top),
        size = Size(w, h),
        style = Stroke(p.thin),
    )

    // animated corner brackets: length breathes subtly with the global phase
    val pulse = 1f + 0.12f * kotlin.math.sin(phase * 2f * Math.PI.toFloat() + t.id)
    val len = (min(w, h) * 0.22f * pulse).coerceIn(p.bracketMin, p.bracketMax)
    val s = p.thick
    // TL
    drawLine(color, Offset(r.left, r.top), Offset(r.left + len, r.top), s)
    drawLine(color, Offset(r.left, r.top), Offset(r.left, r.top + len), s)
    // TR
    drawLine(color, Offset(r.right - len, r.top), Offset(r.right, r.top), s)
    drawLine(color, Offset(r.right, r.top), Offset(r.right, r.top + len), s)
    // BL
    drawLine(color, Offset(r.left, r.bottom - len), Offset(r.left, r.bottom), s)
    drawLine(color, Offset(r.left, r.bottom), Offset(r.left + len, r.bottom), s)
    // BR
    drawLine(color, Offset(r.right - len, r.bottom), Offset(r.right, r.bottom), s)
    drawLine(color, Offset(r.right, r.bottom - len), Offset(r.right, r.bottom), s)

    // label block: "PHONE  98.7%"
    val label = t.label
    val conf = String.format(Locale.US, "%.1f%%", t.confidence * 100f)
    drawIntoCanvas { canvas ->
        val nc = canvas.nativeCanvas
        val labelW = p.labelPaint.measureText(label)
        val confW = p.confPaint.measureText(conf)
        val pad = p.pad
        val textH = p.labelPaint.textSize
        val top = (r.top - textH - pad * 2.6f).coerceAtLeast(p.pad)
        val maxLeft = maxOf(p.pad, size.width - labelW - confW - pad * 4)
        val left = r.left.coerceIn(p.pad, maxLeft)

        p.bgPaint.color = p.panelColor
        nc.drawRect(left - pad, top - pad, left + labelW + confW + pad * 3, top + textH + pad, p.bgPaint)
        // connector tick from label to box
        nc.drawLine(left - pad, top + textH + pad, r.left, r.top, p.tickPaint)

        p.labelPaint.color = (if (focused) Hud.Warning else Hud.Primary).toArgb()
        nc.drawText(label, left, top + textH - p.baselineFix, p.labelPaint)
        nc.drawText(conf, left + labelW + pad * 2, top + textH - p.baselineFix, p.confPaint)
    }

    // tiny technical info: track id
    if (showId) {
        drawIntoCanvas { canvas ->
            canvas.nativeCanvas.drawText(
                String.format(Locale.US, "TRK-%02d", t.id % 100),
                r.right - p.confPaint.measureText("TRK-00"),
                r.bottom + p.confPaint.textSize + p.pad,
                p.idPaint,
            )
        }
    }
}

/** Corner brackets + tick marks on the screen edge - "wearable device" frame. */
private fun DrawScope.drawScreenFrame(p: HudPaints) {
    val m = p.screenMargin
    val len = p.screenBracket
    val c = Hud.PrimaryDim
    val s = p.thin
    // corners
    drawLine(c, Offset(m, m), Offset(m + len, m), s)
    drawLine(c, Offset(m, m), Offset(m, m + len), s)
    drawLine(c, Offset(size.width - m - len, m), Offset(size.width - m, m), s)
    drawLine(c, Offset(size.width - m, m), Offset(size.width - m, m + len), s)
    drawLine(c, Offset(m, size.height - m - len), Offset(m, size.height - m), s)
    drawLine(c, Offset(m, size.height - m), Offset(m + len, size.height - m), s)
    drawLine(c, Offset(size.width - m - len, size.height - m), Offset(size.width - m, size.height - m), s)
    drawLine(c, Offset(size.width - m, size.height - m - len), Offset(size.width - m, size.height - m), s)
    // side tick marks
    val ticks = 5
    for (i in 1..ticks) {
        val y = size.height * i / (ticks + 1f)
        drawLine(Hud.PrimaryFaint, Offset(m, y), Offset(m + p.tickLen, y), s)
        drawLine(Hud.PrimaryFaint, Offset(size.width - m - p.tickLen, y), Offset(size.width - m, y), s)
    }
}

/** Soft horizontal scan band sweeping down the screen. */
private fun DrawScope.drawScanline(phase: Float) {
    val y = size.height * phase
    val band = size.height * 0.06f
    drawRect(
        brush = Brush.verticalGradient(
            colors = listOf(Color.Transparent, Hud.PrimaryFaint, Color.Transparent),
            startY = y - band,
            endY = y + band,
        ),
        topLeft = Offset(0f, y - band),
        size = Size(size.width, band * 2),
    )
}

/** Small rotating reticle at screen center; expands slightly when nothing is locked. */
private fun DrawScope.drawReticle(phase: Float, searching: Boolean) {
    val cx = size.width / 2f
    val cy = size.height / 2f
    val radius = if (searching) 34.dp.toPx() else 24.dp.toPx()
    val sweep = 70f
    rotate(degrees = phase * 360f, pivot = Offset(cx, cy)) {
        for (i in 0 until 3) {
            drawArc(
                color = if (searching) Hud.PrimaryDim else Hud.PrimaryFaint,
                startAngle = i * 120f,
                sweepAngle = sweep,
                useCenter = false,
                topLeft = Offset(cx - radius, cy - radius),
                size = Size(radius * 2, radius * 2),
                style = Stroke(1.5.dp.toPx()),
            )
        }
    }
    drawCircle(Hud.PrimaryDim, radius = 2.dp.toPx(), center = Offset(cx, cy))
}

/** Pre-allocated native paints - zero allocation in the draw loop. */
private class HudPaints(density: Float) {
    val thin = 1.2f * density
    val thick = 2.2f * density
    val pad = 4f * density
    val bracketMin = 10f * density
    val bracketMax = 30f * density
    val screenMargin = 10f * density
    val screenBracket = 26f * density
    val tickLen = 6f * density
    val baselineFix = 2f * density
    val panelColor = Hud.PanelBg.toArgb()

    private val mono: Typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)

    val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = mono
        textSize = 13f * density
        letterSpacing = 0.12f
        color = Hud.Primary.toArgb()
    }
    val confPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.MONOSPACE
        textSize = 11f * density
        color = Hud.TextDim.toArgb()
    }
    val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.MONOSPACE
        textSize = 9f * density
        letterSpacing = 0.1f
        color = Hud.PrimaryDim.toArgb()
    }
    val bgPaint = Paint()
    val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 1f * density
        color = Hud.PrimaryDim.toArgb()
    }
}

