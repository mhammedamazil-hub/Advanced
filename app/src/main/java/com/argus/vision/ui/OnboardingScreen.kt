package com.argus.vision.ui

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.vision.ui.theme.Hud

/**
 * First-launch screen: explains WHY the camera is needed before requesting it,
 * per privacy requirements. Detection is on-device; nothing is uploaded.
 */
@Composable
fun OnboardingScreen(onGranted: () -> Unit) {
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) onGranted() }

    Column(
        Modifier
            .fillMaxSize()
            .background(Hud.Background)
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            "ARGUS",
            color = Hud.Primary, fontSize = 34.sp,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold,
            letterSpacing = 10.sp,
        )
        Text(
            "AUGMENTED RECOGNITION & GUIDANCE SYSTEM",
            color = Hud.TextDim, fontSize = 9.sp,
            fontFamily = FontFamily.Monospace, letterSpacing = 2.sp,
        )
        Spacer(Modifier.height(48.dp))
        Text(
            "CAMERA ACCESS REQUIRED",
            color = Hud.Primary, fontSize = 13.sp,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold,
            letterSpacing = 2.sp,
        )
        Spacer(Modifier.height(16.dp))
        Text(
            "ARGUS analyzes the live camera feed to recognize objects around you.\n\n" +
                "• All detection runs ON THIS DEVICE\n" +
                "• No frames are uploaded anywhere\n" +
                "• Works fully offline / airplane mode\n" +
                "• Microphone stays OFF unless you tap the mic button",
            color = Hud.TextDim, fontSize = 12.sp,
            fontFamily = FontFamily.Monospace, lineHeight = 20.sp,
            textAlign = TextAlign.Start,
        )
        Spacer(Modifier.height(48.dp))
        Text(
            "INITIALIZE VISION SYSTEM",
            color = Hud.Background, fontSize = 13.sp,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold,
            letterSpacing = 2.sp,
            modifier = Modifier
                .clickable { launcher.launch(Manifest.permission.CAMERA) }
                .background(Hud.Primary, RoundedCornerShape(6.dp))
                .border(1.dp, Hud.Primary, RoundedCornerShape(6.dp))
                .padding(horizontal = 24.dp, vertical = 14.dp),
        )
    }
}
