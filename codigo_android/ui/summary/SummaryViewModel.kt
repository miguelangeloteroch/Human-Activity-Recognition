
package com.example.activitydetector.ui.summary

import androidx.lifecycle.ViewModel
import com.example.activitydetector.data.SessionResult
import com.example.activitydetector.ui.listen.ActivityState
import com.example.activitydetector.ui.listen.SegmentEntry
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow


 // ViewModel para la pantalla de resumen de actividad
  //Gestiona el resultado inmutable de la sesión

class SummaryViewModel : ViewModel() {

    private val _summaryState = MutableStateFlow<SummaryUiState>(
        SummaryUiState(
            segments = emptyList(),
            totalActivityMs = 0L,
            walkingMs = 0L,
            joggingMs = 0L
        )
    )
    val summaryState: StateFlow<SummaryUiState> = _summaryState.asStateFlow()

   //Sessionresult inmutable
    fun setSessionResult(sessionResult: SessionResult) {
        _summaryState.value = SummaryUiState(
            segments = sessionResult.segments,
            totalActivityMs = sessionResult.totalActivityMs,
            walkingMs = sessionResult.walkingMs,
            joggingMs = sessionResult.joggingMs
        )
    }
}

//ESTADO DE LA INTERFAZ DE LA PANTALLA SUMMARY
data class SummaryUiState(
    val segments: List<SegmentEntry>,
    val totalActivityMs: Long,
    val walkingMs: Long,
    val joggingMs: Long
) {
    //CALCULAMOS LOS TIEMPOS
    fun formatTime(ms: Long): String {
        val totalSeconds = (ms / 1000).toInt()
        val minutes = totalSeconds / 60
        val seconds = totalSeconds % 60
        return String.format("%02d:%02d", minutes, seconds)
    }

    //OBTENEMOS LA ETIQUETA DE LA ACTIVIDAD
    fun getActivityLabel(activity: ActivityState): String = when (activity) {
        ActivityState.WALKING -> "Andando"
        ActivityState.JOGGING -> "Corriendo"
        ActivityState.STATIONARY -> "Quieto"
        ActivityState.UNKNOWN -> "Detectando"
    }

    //SU RESPECTIVO COLOR
    fun getActivityColor(activity: ActivityState): Long = when (activity) {
        ActivityState.WALKING -> 0xFFFF9F1C  // #FF9F1C
        ActivityState.JOGGING -> 0xFFE63946  // #E63946
        ActivityState.STATIONARY -> 0xFF1F2937  // #1F2937
        ActivityState.UNKNOWN -> 0xFF6B7280  // #6B7280
    }
    
    /**
     * Calcula el porcentaje de walking garantizando que walking% + jogging% = 100%
     * SOURCE OF TRUTH: Se calcula walking%, luego jogging% = 100 - walking%
     */
    fun getWalkingPercentage(): Int {
        if (totalActivityMs <= 0) return 0
        val percentage = (walkingMs * 100.0 / totalActivityMs).toInt()
        return percentage.coerceIn(0, 100)
    }
    
    /**
     * Calcula el porcentaje de jogging como complemento (100 - walking%)
     * Esto garantiza que ambos porcentajes siempre sumen exactamente 100%
     */
    fun getJoggingPercentage(): Int {
        if (totalActivityMs <= 0) return 0
        val walkingPct = getWalkingPercentage()
        return (100 - walkingPct).coerceIn(0, 100)
    }
}
