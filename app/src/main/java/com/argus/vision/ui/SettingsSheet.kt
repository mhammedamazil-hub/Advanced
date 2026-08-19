package com.argus.vision.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.argus.vision.MainViewModel
import com.argus.vision.ui.theme.Hud
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsSheet(vm: MainViewModel, onDismiss: () -> Unit) {
    val settings by vm.settingsStore.settings.collectAsStateWithLifecycle()

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = Hud.Background,
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            SectionTitle("SYSTEM CONFIG")

            SettingSwitch("HUD scanline", settings.scanlineEnabled) {
                vm.settingsStore.update { s -> s.copy(scanlineEnabled = it) }
            }
            SettingSwitch("Track IDs", settings.showTrackIds) {
                vm.settingsStore.update { s -> s.copy(showTrackIds = it) }
            }
            SettingSwitch("Speak answers (TTS)", settings.speakAnswers) {
                vm.settingsStore.update { s -> s.copy(speakAnswers = it) }
            }
            SettingSwitch("Developer / performance panel", settings.showPerfPanel) {
                vm.settingsStore.update { s -> s.copy(showPerfPanel = it) }
            }

            Spacer(Modifier.height(6.dp))
            Text(
                "DETECTION THRESHOLD  ${String.format(Locale.US, "%.0f%%", settings.scoreThreshold * 100)}",
                color = Hud.TextDim, fontSize = 10.sp, fontFamily = FontFamily.Monospace,
            )
            Slider(
                value = settings.scoreThreshold,
                onValueChange = { v -> vm.settingsStore.update { s -> s.copy(scoreThreshold = v) } },
                valueRange = 0.2f..0.8f,
                colors = SliderDefaults.colors(
                    thumbColor = Hud.Primary,
                    activeTrackColor = Hud.PrimaryDim,
                    inactiveTrackColor = Hud.PrimaryFaint,
                ),
            )

            Spacer(Modifier.height(10.dp))
            SectionTitle("CLOUD REASONING (OPTIONAL)")
            Text(
                "Off by default. Object detection is always 100% on-device. When enabled, " +
                    "a single frame is sent to your configured endpoint ONLY when you ask a " +
                    "question local detection can't answer.",
                color = Hud.TextDim, fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                lineHeight = 14.sp,
            )
            SettingSwitch("Enable cloud reasoning", settings.cloudEnabled) {
                vm.settingsStore.update { s -> s.copy(cloudEnabled = it) }
            }
            if (settings.cloudEnabled) {
                CloudField("Endpoint", settings.cloudEndpoint) {
                    vm.settingsStore.update { s -> s.copy(cloudEndpoint = it) }
                }
                CloudField("Model", settings.cloudModel) {
                    vm.settingsStore.update { s -> s.copy(cloudModel = it) }
                }
                CloudField("API key", settings.cloudApiKey, password = true) {
                    vm.settingsStore.update { s -> s.copy(cloudApiKey = it) }
                }
            }

            Spacer(Modifier.height(10.dp))
            SectionTitle("MODELS")
            vm.engine.modelManager.availableModels().forEach { m ->
                Column(Modifier.padding(vertical = 4.dp)) {
                    Text(
                        m.displayName, color = Hud.Primary, fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold,
                    )
                    Text(
                        m.description, color = Hud.TextDim,
                        fontSize = 9.sp, fontFamily = FontFamily.Monospace,
                    )
                }
            }
            Text(
                "Drop custom .tflite detection models into Android/data files/models " +
                    "to add them here.",
                color = Hud.TextDim, fontSize = 9.sp, fontFamily = FontFamily.Monospace,
            )
            Spacer(Modifier.height(28.dp))
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(
        text,
        color = Hud.Primary,
        fontSize = 12.sp,
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Bold,
        letterSpacing = 2.sp,
        modifier = Modifier.padding(vertical = 8.dp),
    )
}

@Composable
private fun SettingSwitch(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            label, color = Hud.TextDim, fontSize = 11.sp,
            fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
        )
        Switch(
            checked = checked,
            onCheckedChange = onChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Hud.Primary,
                checkedTrackColor = Hud.PrimaryFaint,
                uncheckedThumbColor = Hud.TextDim,
                uncheckedTrackColor = Hud.PanelBg,
            ),
        )
    }
}

@Composable
private fun CloudField(
    label: String,
    value: String,
    password: Boolean = false,
    onChange: (String) -> Unit,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label, fontSize = 10.sp, fontFamily = FontFamily.Monospace) },
        singleLine = true,
        visualTransformation = if (password) {
            androidx.compose.ui.text.input.PasswordVisualTransformation()
        } else {
            androidx.compose.ui.text.input.VisualTransformation.None
        },
        textStyle = androidx.compose.ui.text.TextStyle(
            fontFamily = FontFamily.Monospace, fontSize = 12.sp,
        ),
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
    )
}
