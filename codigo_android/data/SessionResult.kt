package com.example.activitydetector.data

import com.example.activitydetector.ui.listen.SegmentEntry


data class SessionResult(
    val segments: List<SegmentEntry>,
    val totalActivityMs: Long,
    val walkingMs: Long,
    val joggingMs: Long,
    val numberOfIntervals: Int,
    val endReason: EndReason
) {
    enum class EndReason {
        STATIONARY,         // Detectado automáticamente, postmovimiento
        TIMEOUT_LISTENING   // Timeout de 30s de inicio. No se detecta actividad en ningun momento
    }
    
//Llevamos de los ms a segudnos, y de los segundos pasamos a min +segs
    fun formatTime(ms: Long): String {
        val totalSeconds = (ms / 1000).toInt()
        val minutes = totalSeconds / 60
        val seconds = totalSeconds % 60
        return String.format("%02d:%02d", minutes, seconds)
    }
//Calculamos el porcentaje de walking, y luego restamos pa conseguir el de jogging
//Así, garantizamos que siempre nos salga 100%

    fun getWalkingPercentage(): Int {
        if (totalActivityMs <= 0) return 0
        val percentage = (walkingMs * 100.0 / totalActivityMs).toInt()
        return percentage.coerceIn(0, 100)
    }

    fun getJoggingPercentage(): Int {
        if (totalActivityMs <= 0) return 0
        val walkingPct = getWalkingPercentage()
        return (100 - walkingPct).coerceIn(0, 100)
    }
}

