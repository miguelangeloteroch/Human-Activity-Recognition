package com.example.activitydetector.ui.listen

import android.app.Application
import android.hardware.SensorManager
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.activitydetector.sensor.SensorController
import com.example.activitydetector.ml.HarDecisionConfig
import com.example.activitydetector.ml.MajorityVoteHarDecisionEngine
import com.example.activitydetector.ml.InferencePreprocessorWithNormalization
import com.example.activitydetector.ml.Tflite
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * Estado de actividad detectada
 */
enum class ActivityState {
    WALKING,
    JOGGING,
    STATIONARY,
    UNKNOWN
}

/**
 * Entrada del historial de segmentos
 */
data class SegmentEntry(
    val activity: ActivityState,
    val durationMs: Long,
    val endedAtEpochMs: Long
)

/**
 * ViewModel para la pantalla de escucha/detección
 * Integra el pipeline completo HAR:
 * - Resampling a 20 Hz (InferencePreprocessor)
 * - Inferencia TFLite (cada 10 ticks = 0.5s)
 * - Decisión estable (HarDecisionEngine)
 * - Actualización de UI
 */
class ListenViewModel(application: Application) : AndroidViewModel(application) {
    // Sensor y controlador
    private val sensorManager: SensorManager? = application.getSystemService(Application.SENSOR_SERVICE) as? SensorManager
    private val sensorController: SensorController? = sensorManager?.let { SensorController(it) }

    // Pipeline HAR
    private val preprocessor = InferencePreprocessorWithNormalization(application)
    private val tflite = Tflite(application)
    private val harEngine = MajorityVoteHarDecisionEngine(HarDecisionConfig())

    // Contador para step=10 (ejecutar predict cada 10 ticks de 20Hz ≈ 0.5s)
    private var tickCounter = 0

    // Estado actual de actividad
    private var currentActivity: ActivityState = ActivityState.UNKNOWN
    private var segmentStartTimeMs: Long = 0L

    // Umbral mínimo para guardar segmentos (2 segundos)
    private val MIN_SEGMENT_MS = 2000L

    // Resultado final de la sesión (snapshot inmutable)
    var lastSessionResult: com.example.activitydetector.data.SessionResult? = null
        private set

    // Flag para evitar múltiples finalizaciones
    private var hasFinalizedSession = false

    // Flag para detener el timer después de finalizar
    private var isSessionActive = true

    // Callback para fin de actividad
    var onActivityFinished: () -> Unit = {}

    private val _uiState = MutableStateFlow(
        ListenUiState(
            currentActivity = ActivityState.UNKNOWN,
            currentSegmentElapsedMs = 0L,
            segmentsHistory = emptyList(),
            isDetecting = false,
            userAlias = null,
            listeningSecondsLeft = 30,
            isListeningCountdownActive = false
        )
    )
    val uiState: StateFlow<ListenUiState> = _uiState.asStateFlow()

    private var motionFlowJob: kotlinx.coroutines.Job? = null
    private var segmentTimerJob: kotlinx.coroutines.Job? = null
    private var countdownJob: kotlinx.coroutines.Job? = null

    init {
        if (sensorController != null) {
            startListening()
        } else {
            _uiState.update { it.copy(isDetecting = false) }
        }
    }

  //Actualizamos el alias del usuario para mostrarlo
    fun setUserAlias(alias: String?) {
        _uiState.update { it.copy(userAlias = alias) }
    }

    //Función startListening
    private fun startListening() {
        _uiState.update { it.copy(isDetecting = true) }
        harEngine.reset()
        tickCounter = 0 //Iniciamos el contador de ticks a 0
        currentActivity = ActivityState.UNKNOWN //La actividad inicial es el estado DESCONOCIDO
        segmentStartTimeMs = SystemClock.elapsedRealtime()
        isSessionActive = true  // Resetear flag para nueva sesión
        hasFinalizedSession = false  // Resetear flag de finalización

        // Iniciar cuenta atrás de 30s para detectar actividad
        startListeningCountdown()

        motionFlowJob = viewModelScope.launch {
            try {
                val accelFlow = sensorController?.accelerometerFlow()
                if (accelFlow == null) {
                    _uiState.update { it.copy(isDetecting = false) }
                    return@launch
                }

                accelFlow.collect { sensorData ->
                    // sensorData: FloatArray [x, y, z]
                    val x = sensorData[0]
                    val y = sensorData[1]
                    val z = sensorData[2]
                    val timestampMs = SystemClock.elapsedRealtime()

                    // PASO 1: Resamplemosa 20Hz y normalizamos
                    val pushed = preprocessor.processAccelerometerEvent(timestampMs, x, y, z)
                    tickCounter += pushed

                    // Verificamos si la ventana está lista y si por otra parte alcanzamos step=10
                    if (preprocessor.hasWindow() && tickCounter >= 10) {
                        tickCounter -= 10

                        // En caso positivo, se dará la INFERENCIA
                        // Obtenemos ventana de 50 muestras normalizadas
                        val window = preprocessor.getWindow()  // Array<FloatArray>[50][4]

                        // PASO 4: Inferencia TFLite
                        val inputArray = arrayOf(window)  // Envolver en lote [1][50][4]
                        val probs = tflite.predict(inputArray)[0]  // [3] (jogging, stationary, walking)

                        // PASO 5: Decisión estable
                        val decision = harEngine.update(timestampMs, probs)

                        // PASO 6: Actualizar UI
                        updateActivityState(decision.stableLabel)
                    }
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(isDetecting = false) }
            }
        }

        // Iniciar ticker del cronómetro de segmento
        startSegmentTimer()
    }
    
    private fun updateActivityState(stableLabel: String) {
        val newActivity = when (stableLabel) {
            "walking" -> ActivityState.WALKING
            "jogging" -> ActivityState.JOGGING
            "stationary" -> ActivityState.STATIONARY
            else -> ActivityState.UNKNOWN
        }
        
        onActivityChanged(newActivity)
    }
    
    /**
     * Maneja el cambio de estado de actividad.
     * Cierra el segmento anterior y resetea el cronómetro.
     * 
     * LÓGICA ESPECIAL PARA UNKNOWN (primeros 30s):
     * - Si detecta WALKING/JOGGING → cancela countdown y comienza actividad
     * - Si detecta STATIONARY → IGNORA (no termina sesión hasta timeout de 30s)
     * 
     * LÓGICA DESPUÉS DE SALIR DE UNKNOWN:
     * - STATIONARY termina la sesión normalmente
     */
    private fun onActivityChanged(newState: ActivityState) {
        if (newState == currentActivity) {
            // No hay cambio, solo actualizar el tiempo transcurrido
            return
        }
        
        // CASO 1: Estamos en UNKNOWN (countdown activo) y detectamos STATIONARY
        // → IGNORAR: no terminar sesión, esperar a que termine el countdown de 30s
        if (currentActivity == ActivityState.UNKNOWN && 
            _uiState.value.isListeningCountdownActive && 
            newState == ActivityState.STATIONARY) {
            // No hacer nada, mantener estado UNKNOWN y dejar que el countdown siga
            return
        }
        
        // CASO 2: Detectamos WALKING o JOGGING
        // → Cancelar countdown y comenzar actividad normal
        if (newState == ActivityState.WALKING || newState == ActivityState.JOGGING) {
            cancelListeningCountdown()
            // Continuar con el flujo normal de cambio de estado (abajo)
        }
        
        // CASO 3: Detectamos STATIONARY cuando YA HAY actividad real (no en UNKNOWN)
        // → Terminar la sesión inmediatamente
        if (newState == ActivityState.STATIONARY && !_uiState.value.isListeningCountdownActive) {
            finishSession(com.example.activitydetector.data.SessionResult.EndReason.STATIONARY)
            return
        }
        
        val currentTime = SystemClock.elapsedRealtime()
        val elapsedMs = currentTime - segmentStartTimeMs
        
        // Cerrar segmento anterior si es válido (solo si NO vamos a STATIONARY)
        if (currentActivity != ActivityState.UNKNOWN && elapsedMs >= MIN_SEGMENT_MS) {
            val segment = SegmentEntry(
                activity = currentActivity,
                durationMs = elapsedMs,
                endedAtEpochMs = System.currentTimeMillis()
            )
            
            _uiState.update { state ->
                val newHistory = listOf(segment) + state.segmentsHistory
                state.copy(segmentsHistory = newHistory)
            }
        }
        
        // Cambiar al nuevo estado y resetear cronómetro
        currentActivity = newState
        segmentStartTimeMs = currentTime
        
        _uiState.update { 
            it.copy(
                currentActivity = newState,
                currentSegmentElapsedMs = 0L
            )
        }
    }
    
    /**
     * Ticker del cronómetro de segmento (actualiza cada 200ms)
     */
    private fun startSegmentTimer() {
        segmentTimerJob?.cancel()
        segmentTimerJob = viewModelScope.launch {
            while (isSessionActive) {
                delay(200)
                if (!isSessionActive) break // Guard adicional

                val currentTime = SystemClock.elapsedRealtime()
                val elapsedMs = currentTime - segmentStartTimeMs
                
                _uiState.update { 
                    it.copy(currentSegmentElapsedMs = elapsedMs)
                }
            }
        }
    }
    
    /**
     * Inicia cuenta atrás de 30 segundos para que el usuario comience la actividad.
     * Si llega a 0 sin detectar WALKING/JOGGING, finaliza la sesión automáticamente.
     */
    private fun startListeningCountdown() {
        countdownJob?.cancel()
        _uiState.update { 
            it.copy(
                listeningSecondsLeft = 30,
                isListeningCountdownActive = true
            )
        }
        
        countdownJob = viewModelScope.launch {
            repeat(30) { iteration ->
                delay(1000L)
                val secondsLeft = 30 - iteration - 1
                _uiState.update { it.copy(listeningSecondsLeft = secondsLeft) }
                
                // Si llegamos a 0, finalizar sesión por timeout
                if (secondsLeft == 0) {
                    finishSession(com.example.activitydetector.data.SessionResult.EndReason.TIMEOUT_LISTENING)
                }
            }
        }
    }
    
    /**
     * Cancela la cuenta atrás de listening (llamar cuando se detecta actividad real)
     */
    private fun cancelListeningCountdown() {
        countdownJob?.cancel()
        countdownJob = null
        _uiState.update { 
            it.copy(isListeningCountdownActive = false)
        }
    }
    
    /**
     * Finaliza la sesión de forma controlada:
     * 1. CONGELA el tiempo inmediatamente (isSessionActive = false)
     * 2. Captura el último segmento activo con tiempo exacto
     * 3. Crea SessionResult inmutable con segmentos finales
     * 4. Detiene sensores y cancela jobs
     * 5. Notifica al callback
     * 
     * Esta función es idempotente: solo se ejecuta una vez gracias al flag hasFinalizedSession
     */
    private fun finishSession(endReason: com.example.activitydetector.data.SessionResult.EndReason = com.example.activitydetector.data.SessionResult.EndReason.STATIONARY) {
        // Guard: evitar múltiples finalizaciones
        if (hasFinalizedSession) {
            return
        }
        hasFinalizedSession = true
        
        // 1. PASO CRÍTICO: Congelar el tiempo INMEDIATAMENTE
        // Esto evita que el timer siga actualizando currentSegmentElapsedMs
        isSessionActive = false

        // 2. Capturar el tiempo EXACTO del último evento de sensado
        // El preprocesador tiene el timestamp más reciente (en ms)
        val freezeTimeMs = SystemClock.elapsedRealtime()
        val elapsedMsLastSegment = freezeTimeMs - segmentStartTimeMs

        // 3. Recopilar TODOS los segmentos ANTES de crear el SessionResult
        // Primero: segmentos históricos
        var allSegments = _uiState.value.segmentsHistory.toMutableList()

        // Segundo: añadir último segmento activo si existe
        if ((currentActivity == ActivityState.WALKING || currentActivity == ActivityState.JOGGING)
            && elapsedMsLastSegment >= 1000L) {
            val lastSegment = SegmentEntry(
                activity = currentActivity,
                durationMs = elapsedMsLastSegment,  // Usar tiempo congelado exacto
                endedAtEpochMs = System.currentTimeMillis()
            )
            allSegments = (listOf(lastSegment) + allSegments).toMutableList()
        }
        
        // 4. Detener sensores ANTES de crear SessionResult
        sensorController?.stop()
        
        // 5. Cancelar jobs activos
        motionFlowJob?.cancel()
        segmentTimerJob?.cancel()
        countdownJob?.cancel()
        motionFlowJob = null
        segmentTimerJob = null
        countdownJob = null
        
        // 6. Crear SessionResult inmutable con los segmentos recopilados
        lastSessionResult = createSessionResult(allSegments.toList(), endReason)

        // 7. Actualizar estado a no detectando
        _uiState.update { it.copy(isDetecting = false) }
        
        // 8. Notificar callback con el resultado
        onActivityFinished()
    }
    
    /**
     * Crea un SessionResult inmutable a partir de los segmentos
     * 
     * SOURCE OF TRUTH: totalActivityMs = walkingMs + joggingMs
     * Solo se cuentan actividades válidas (WALKING/JOGGING).
     * STATIONARY y UNKNOWN no se incluyen en el total.
     */
    private fun createSessionResult(
        segments: List<SegmentEntry>,
        endReason: com.example.activitydetector.data.SessionResult.EndReason = com.example.activitydetector.data.SessionResult.EndReason.STATIONARY
    ): com.example.activitydetector.data.SessionResult {
        var walkingMs = 0L
        var joggingMs = 0L
        
        segments.forEach { segment ->
            when (segment.activity) {
                ActivityState.WALKING -> walkingMs += segment.durationMs
                ActivityState.JOGGING -> joggingMs += segment.durationMs
                else -> {} // STATIONARY y UNKNOWN no cuentan en total de actividad
            }
        }
        
        // Total es la suma exacta de walking + jogging (source of truth)
        val totalMs = walkingMs + joggingMs
        
        return com.example.activitydetector.data.SessionResult(
            segments = segments.toList(), // Copia inmutabl
            walkingMs = walkingMs,
            joggingMs = joggingMs,
            totalActivityMs = walkingMs+joggingMs,
            numberOfIntervals = segments.size,
            endReason = endReason
        )
    }
    
    /**
     * Guarda el segmento actual si es una actividad válida (WALKING o JOGGING)
     * incluso si no alcanza el mínimo de 2 segundos.
     * Se usa cuando se termina la actividad para asegurar que se guarde el último segmento.
     */
    fun saveCurrentSegmentIfActive() {
        val currentTime = SystemClock.elapsedRealtime()
        val elapsedMs = currentTime - segmentStartTimeMs
        
        // Guardar si es WALKING o JOGGING y tiene al menos 1 segundo
        if ((currentActivity == ActivityState.WALKING || currentActivity == ActivityState.JOGGING) 
            && elapsedMs >= 1000L) {
            val segment = SegmentEntry(
                activity = currentActivity,
                durationMs = elapsedMs,
                endedAtEpochMs = System.currentTimeMillis()
            )
            
            _uiState.update { state ->
                val newHistory = listOf(segment) + state.segmentsHistory
                state.copy(segmentsHistory = newHistory)
            }
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        sensorController?.stop()
        motionFlowJob?.cancel()
        segmentTimerJob?.cancel()
        countdownJob?.cancel()
    }
}

/**
 * Estado de la UI de la pantalla de escucha
 */
data class ListenUiState(
    val currentActivity: ActivityState,
    val currentSegmentElapsedMs: Long,
    val segmentsHistory: List<SegmentEntry>,
    val isDetecting: Boolean,
    val userAlias: String? = null,
    val listeningSecondsLeft: Int = 30,
    val isListeningCountdownActive: Boolean = false
) {
    /**
     * Formatea el tiempo del segmento actual en mm:ss
     */
    val formattedSegmentTime: String
        get() {
            val totalSeconds = (currentSegmentElapsedMs / 1000).toInt()
            val minutes = totalSeconds / 60
            val seconds = totalSeconds % 60
            return String.format("%02d:%02d", minutes, seconds)
        }
    
    /**
     * Obtiene el color de fondo según la actividad
     */
    val backgroundColor: Long
        get() = when (currentActivity) {
            ActivityState.WALKING -> 0xFFFF9F1C  // #FF9F1C
            ActivityState.JOGGING -> 0xFFE63946  // #E63946
            ActivityState.STATIONARY -> 0xFF1F2937  // #1F2937
            ActivityState.UNKNOWN -> 0xFF1F2937  // #1F2937 (gris oscuro)
        }

    //texto de la actividad,
    val activityDisplayText: String
        get() = when (currentActivity) {
            ActivityState.WALKING -> "ANDANDO"
            ActivityState.JOGGING -> "CORRIENDO"
            ActivityState.STATIONARY -> "QUIETO"
            ActivityState.UNKNOWN -> ""
        }}

