package com.argus.vision.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ArgusColors = darkColorScheme(
    primary = Hud.Primary,
    secondary = Hud.Accent,
    background = Hud.Background,
    surface = Hud.Background,
    onPrimary = Hud.Background,
    onBackground = Hud.Primary,
    onSurface = Hud.Primary,
)

@Composable
fun ArgusTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = ArgusColors, content = content)
}
