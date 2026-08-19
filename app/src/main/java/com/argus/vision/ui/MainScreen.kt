package com.argus.vision.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.argus.vision.MainViewModel
import com.argus.vision.camera.CameraController
import com.argus.vision.hud.HudOverlay
import com.argus.vision.settings.DetectionMode
import com.argus.vision.ui.theme.Hud
import com.argus.vision.vision.EngineState
import com.argus.vision.voice.VoiceState
import java.util.Locale

@Composable
fun MainScreen(vm: MainViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val engineState by vm.engine.state.collectAsStateWithLifecycle()
    val tracks by vm.engine.tracks.collectAsStateWithLifecycle()
    val frameSize by vm.engine.frameSize.collectAsStateWithLifecycle()
    val perf by vm.engine.perf.collectAsStateWithLifecycle()
    val system by vm.system.collectAsStateWithLifecycle()
    val settings by vm.settingsStore.settings.collectAsStateWithLifecycle()
    val voiceState by vm.voiceState.collectAsStateWithLifecycle()
    val assistant by vm.assistant.collectAsStateWithLifecycle()
    val focusId by vm.focusTrackId.collectAsStateWithLifecycle()

    var showSettings by remember { mutableStateOf(false) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val cameraController = remember { CameraController(context.applicationContext) }
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    LaunchedEffect(Unit) {
        try {
            cameraController.start(lifecycleOwner, previewView, vm.engine::onFrame)
        } catch (t: Throwable) {
            cameraError = t.message ?: "Camera failed to start"
        }
    }
    DisposableEffect(Unit) {
        onDispose { cameraController.shutdown() }
    }

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) vm.onMicPressed() }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        HudOverlay(
            tracks = tracks,
            frameWidth = frameSize.first,
            frameHeight = frameSize.second,
            focusTrackId = focusId,
            showTrackIds = settings.showTrackIds,
            scanlineEnabled = settings.scanlineEnabled,
            modifier = Modifier.fillMaxSize(),
        )

        // ---------- top status bar ----------
        Row(
            Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 18.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            StatusChip(engineState)
            Spacer(Modifier.weight(1f))
            HudText("OBJ ${tracks.size}", dim = true)
            Spacer(Modifier.width(12.dp))
            HudText(String.format(Locale.US, "%02.0f FPS", perf.cameraFps), dim = true)
            Spacer(Modifier.width(12.dp))
            Text(
                "⚙",
                color = Hud.PrimaryDim,
                fontSize = 18.sp,
                modifier = Modifier.clickable { showSettings = true },
            )
        }

        // ---------- engine error banner (camera keeps running) ----------
        if (engineState is EngineState.Error || cameraError != null) {
            Column(
                Modifier
                    .align(Alignment.TopCenter)
                    .statusBarsPadding()
                    .padding(top = 56.dp, start = 24.dp, end = 24.dp)
                    .clip(RoundedCornerShape(6.dp))
                    .background(Hud.PanelBg)
                    .border(1.dp, Hud.Danger.copy(alpha = 0.6f), RoundedCornerShape(6.dp))
                    .padding(12.dp),
            ) {
                Text(
                    cameraError ?: (engineState as EngineState.Error).message,
                    color = Hud.Danger,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    textAlign = TextAlign.Center,
                )
            }
        }

        // ---------- performance / developer panel ----------
        if (settings.showPerfPanel) {
            Column(
                Modifier
                    .align(Alignment.TopStart)
                    .statusBarsPadding()
                    .padding(start = 18.dp, top = 52.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(Hud.PanelBg)
                    .padding(horizontal = 10.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                PerfLine("CAM", String.format(Locale.US, "%.1f fps  %dx%d", perf.cameraFps, perf.frameWidth, perf.frameHeight))
                PerfLine("INF", String.format(Locale.US, "%.1f fps  %.0f ms", perf.inferenceFps, perf.inferenceLatencyMs))
                PerfLine("MDL", "${perf.modelName} [${perf.delegate}]")
                PerfLine("CPU", String.format(Locale.US, "%.1f %%", system.cpuPercent))
                PerfLine("MEM", "${system.usedMemMb} MB / ${system.totalMemMb} MB")
                PerfLine("THM", perf.thermal)
                PerfLine("OBJ", "${perf.objectCount} tracked")
            }
        }

        // ---------- assistant answer overlay ----------
        AnimatedVisibility(
            visible = assistant != null,
            enter = fadeIn(), exit = fadeOut(),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 148.dp, start = 24.dp, end = 24.dp),
        ) {
            val msg = assistant
            if (msg != null) {
                Column(
                    Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(Hud.PanelBg)
                        .border(1.dp, Hud.PrimaryFaint, RoundedCornerShape(8.dp))
                        .padding(14.dp),
                ) {
                    Text(
                        "> ${msg.query.uppercase(Locale.US)}",
                        color = Hud.TextDim, fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace, letterSpacing = 1.sp,
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        msg.answer,
                        color = Hud.Primary, fontSize = 14.sp,
                        fontFamily = FontFamily.Monospace, lineHeight = 20.sp,
                    )
                }
            }
        }

        // ---------- bottom controls ----------
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(bottom = 18.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (voiceState is VoiceState.Listening) {
                HudText("LISTENING...", dim = false)
                Spacer(Modifier.height(8.dp))
            } else if (voiceState is VoiceState.Error) {
                Text(
                    (voiceState as VoiceState.Error).message.uppercase(Locale.US),
                    color = Hud.Warning, fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                )
                Spacer(Modifier.height(8.dp))
            }

            MicButton(listening = voiceState is VoiceState.Listening) {
                val granted = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.RECORD_AUDIO
                ) == PackageManager.PERMISSION_GRANTED
                if (granted) vm.onMicPressed()
                else micLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }

            Spacer(Modifier.height(14.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                DetectionMode.entries.forEach { mode ->
                    ModeChip(
                        label = mode.name,
                        selected = settings.mode == mode,
                        onClick = { vm.settingsStore.update { it.copy(mode = mode) } },
                    )
                }
            }
        }
    }

    if (showSettings) {
        SettingsSheet(vm = vm, onDismiss = { showSettings = false })
    }
}

@Composable
private fun StatusChip(state: EngineState) {
    val transition = rememberInfiniteTransition(label = "status")
    val blink by transition.animateFloat(
        0.4f, 1f,
        infiniteRepeatable(tween(900), RepeatMode.Reverse),
        label = "blink",
    )
    val (text, color) = when (state) {
        is EngineState.Online -> "VISION ONLINE" to Hud.Primary
        is EngineState.Initializing -> "LOADING MODEL" to Hud.Warning
        is EngineState.Error -> "VISION OFFLINE" to Hud.Danger
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier
                .size(7.dp)
                .alpha(if (state is EngineState.Online) 1f else blink)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(Modifier.width(8.dp))
        Text(
            text,
            color = color,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            letterSpacing = 2.sp,
        )
    }
}

@Composable
private fun HudText(text: String, dim: Boolean) {
    Text(
        text,
        color = if (dim) Hud.TextDim else Hud.Primary,
        fontSize = 11.sp,
        fontFamily = FontFamily.Monospace,
        letterSpacing = 1.sp,
    )
}

@Composable
private fun PerfLine(key: String, value: String) {
    Row {
        Text(
            key, color = Hud.PrimaryDim, fontSize = 9.sp,
            fontFamily = FontFamily.Monospace, modifier = Modifier.width(34.dp),
        )
        Text(value, color = Hud.TextDim, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun MicButton(listening: Boolean, onClick: () -> Unit) {
    val transition = rememberInfiniteTransition(label = "mic")
    val pulse by transition.animateFloat(
        0f, 1f,
        infiniteRepeatable(tween(1200), RepeatMode.Restart),
        label = "pulse",
    )
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(68.dp)
            .clip(CircleShape)
            .background(if (listening) Hud.Primary.copy(alpha = 0.18f) else Hud.PanelBg)
            .border(1.5.dp, if (listening) Hud.Primary else Hud.PrimaryDim, CircleShape)
            .clickable(onClick = onClick),
    ) {
        if (listening) {
            androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) {
                drawCircle(
                    color = Hud.Primary.copy(alpha = (1f - pulse) * 0.5f),
                    radius = size.minDimension / 2f * (0.6f + pulse * 0.45f),
                    center = Offset(size.width / 2, size.height / 2),
                    style = Stroke(2.dp.toPx()),
                )
            }
        }
        // simple mic glyph drawn with shapes (no icon dependency)
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier
                    .size(width = 14.dp, height = 22.dp)
                    .clip(RoundedCornerShape(7.dp))
                    .background(if (listening) Hud.Primary else Hud.PrimaryDim)
            )
            Spacer(Modifier.height(3.dp))
            Box(
                Modifier
                    .size(width = 22.dp, height = 2.dp)
                    .background(if (listening) Hud.Primary else Hud.PrimaryDim)
            )
        }
    }
}

@Composable
private fun ModeChip(label: String, selected: Boolean, onClick: () -> Unit) {
    Text(
        label,
        color = if (selected) Hud.Background else Hud.PrimaryDim,
        fontSize = 10.sp,
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Bold,
        letterSpacing = 1.sp,
        modifier = Modifier
            .clip(RoundedCornerShape(4.dp))
            .background(if (selected) Hud.Primary else Hud.PanelBg)
            .border(1.dp, if (selected) Hud.Primary else Hud.PrimaryFaint, RoundedCornerShape(4.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    )
}
