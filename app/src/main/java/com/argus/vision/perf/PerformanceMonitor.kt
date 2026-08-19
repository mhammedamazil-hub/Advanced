package com.argus.vision.perf

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import android.os.SystemClock

/** Snapshot of runtime performance for the developer panel. */
data class PerfStats(
    val cameraFps: Float = 0f,
    val inferenceFps: Float = 0f,
    val inferenceLatencyMs: Float = 0f,
    val modelName: String = "-",
    val delegate: String = "-",
    val objectCount: Int = 0,
    val thermal: String = "-",
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
)

/** System-level metrics sampled on demand (cheap, no background polling). */
class PerformanceMonitor(context: Context) {

    private val activityManager =
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    private var lastCpuTime = Process.getElapsedCpuTime()
    private var lastWallTime = SystemClock.uptimeMillis()
    private val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    /** App CPU usage in % of total device CPU since the last call. */
    fun sampleCpuPercent(): Float {
        val cpu = Process.getElapsedCpuTime()
        val wall = SystemClock.uptimeMillis()
        val dCpu = cpu - lastCpuTime
        val dWall = (wall - lastWallTime).coerceAtLeast(1)
        lastCpuTime = cpu
        lastWallTime = wall
        return (dCpu.toFloat() / (dWall * cores)) * 100f
    }

    /** (usedMb, totalDeviceMb) */
    fun sampleMemoryMb(): Pair<Int, Int> {
        val heap = Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }
        val native = Debug.getNativeHeapAllocatedSize()
        val info = ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }
        val usedMb = ((heap + native) / (1024 * 1024)).toInt()
        val totalMb = (info.totalMem / (1024 * 1024)).toInt()
        return usedMb to totalMb
    }
}
