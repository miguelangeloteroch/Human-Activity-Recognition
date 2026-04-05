package com.example.activitydetector.ml

import android.util.Log
import java.util.Deque
import java.util.LinkedList

//AQUÍ SE IMPLEMENTA EL MOTOR DE DECISIÓN

/*
  Lógica de decisión (6 pasos):
 1. A historySize < N → UNKNOWN
 2. Raw prediction: argmax(probs)
 3.  Si confidence < minConfidence, se mantiene el estado anterior
 4. Historial FIFO: almacenar tuplas (label, confidence) de tamaño N
 5. Majority vote: calcular moda; si empate → RECHAZAR
 6. Para satationary:
     - Mayoría estricta: count >= 3 (de 5)
     - Confianza promedio: mean(confidence) >= 0.65
     - No aceptar por tie-break

  Two-hit switch (walking ↔ jogging):
  - Si rawLabel != stableLabel (y ninguno es "stationary")
  - Y confidence >= switchConfidence
  - Y ocurre durante switchConsecutiveHits updates consecutivos
  - Entonces cambiar inmediatamente sin esperar majority vote

 Histéresis: nunca retroceder a UNKNOWN una vez que hay estado estable
 */
class MajorityVoteHarDecisionEngine(
    private val config: HarDecisionConfig
) : HarDecisionEngine {

    // Historial de predicciones: tuplas (label, confidence)
    private val predictionHistory: Deque<Pair<String, Float>> = LinkedList()

    // Último estado estable consolidado (para histéresis)
    private var lastStableLabel: String? = null
    private var lastStableIndex: Int? = null

    // Estado para "two-hit switch" (cambio rápido walking↔jogging)
    private var switchTargetLabel: String? = null
    private var switchConsecutiveCount: Int = 0

    override fun reset() {
        predictionHistory.clear()
        lastStableLabel = null
        lastStableIndex = null
        switchTargetLabel = null
        switchConsecutiveCount = 0
    }

    override fun update(timestampMs: Long, probs: FloatArray): HarDecisionOutput {
        // PASO 2: Calcular predicción cruda (argmax)
        val rawIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val rawLabel = config.classes[rawIndex]
        val confidence = probs[rawIndex]

        // PASO 3: Filtro de baja confianza
        if (confidence < config.minConfidence) {
            // Mantener estado anterior (histéresis) o UNKNOWN si no hay previo
            val currentStableLabel = lastStableLabel ?: "UNKNOWN"
            val currentStableIndex = lastStableIndex
            val isStable = lastStableLabel != null

            // Reiniciar contador de switch por baja confianza
            switchTargetLabel = null
            switchConsecutiveCount = 0

            // Logging de verificación
            Log.d("HAR_ENGINE", "ts=$timestampMs raw=$rawLabel (${String.format("%.2f", confidence)}) stable=$currentStableLabel history=${predictionHistory.size}")

            return HarDecisionOutput(
                timestampMs = timestampMs,
                probs = probs,
                rawIndex = rawIndex,
                rawLabel = rawLabel,
                confidence = confidence,
                stableLabel = currentStableLabel,
                stableIndex = currentStableIndex,
                isStable = isStable,
                historySize = predictionHistory.size,
                stableWindowSize = config.stableWindowSize
            )
        }

        // TWO-HIT SWITCH: Verificar cambio rápido walking↔jogging
        val currentStableLabel = lastStableLabel ?: "UNKNOWN"
        val shouldCheckSwitch = lastStableLabel != null && 
                                rawLabel != currentStableLabel &&
                                rawLabel != "stationary" && 
                                currentStableLabel != "stationary" &&
                                confidence >= config.switchConfidence

        if (shouldCheckSwitch) {
            // Verificar si es el mismo target que el update anterior
            if (switchTargetLabel == rawLabel) {
                switchConsecutiveCount++
            } else {
                // Nuevo target, reiniciar contador
                switchTargetLabel = rawLabel
                switchConsecutiveCount = 1
            }

            // Si alcanzamos el umbral, hacer switch inmediato
            if (switchConsecutiveCount >= config.switchConsecutiveHits) {
                val newStableIndex = config.classes.indexOf(rawLabel).takeIf { it >= 0 }
                lastStableLabel = rawLabel
                lastStableIndex = newStableIndex

                // Reiniciar contador después del switch
                switchTargetLabel = null
                switchConsecutiveCount = 0

                // Añadir al historial para mantener consistencia
                predictionHistory.addLast(Pair(rawLabel, confidence))
                if (predictionHistory.size > config.stableWindowSize) {
                    predictionHistory.removeFirst()
                }

                // Logging de verificación
                Log.d("HAR_ENGINE", "ts=$timestampMs raw=$rawLabel (${String.format("%.2f", confidence)}) stable=$rawLabel history=${predictionHistory.size}")

                return HarDecisionOutput(
                    timestampMs = timestampMs,
                    probs = probs,
                    rawIndex = rawIndex,
                    rawLabel = rawLabel,
                    confidence = confidence,
                    stableLabel = rawLabel,
                    stableIndex = newStableIndex,
                    isStable = true,
                    historySize = predictionHistory.size,
                    stableWindowSize = config.stableWindowSize
                )
            }
        } else {
            // Condiciones no se cumplen, reiniciar contador
            switchTargetLabel = null
            switchConsecutiveCount = 0
        }

        // PASO 4: Añadir predicción al historial (FIFO)
        predictionHistory.addLast(Pair(rawLabel, confidence))
        if (predictionHistory.size > config.stableWindowSize) {
            predictionHistory.removeFirst()
        }

        // PASO 1 + 5 + 6: Calcular decisión estable
        val (stableLabel, stableIndex) = calculateStableDecision()

        // Histéresis: actualizar último estado estable consolidado
        if (stableLabel != "UNKNOWN") {
            lastStableLabel = stableLabel
            lastStableIndex = config.classes.indexOf(stableLabel).takeIf { it >= 0 }
        }

        // Logging de verificación
        Log.d("HAR_ENGINE", "ts=$timestampMs raw=$rawLabel (${String.format("%.2f", confidence)}) stable=$stableLabel history=${predictionHistory.size}")

        return HarDecisionOutput(
            timestampMs = timestampMs,
            probs = probs,
            rawIndex = rawIndex,
            rawLabel = rawLabel,
            confidence = confidence,
            stableLabel = stableLabel,
            stableIndex = stableIndex,
            isStable = stableLabel != "UNKNOWN",
            historySize = predictionHistory.size,
            stableWindowSize = config.stableWindowSize
        )
    }

    private fun calculateStableDecision(): Pair<String, Int?> {
        // PASO 1: Arranque - historial no está lleno
        if (predictionHistory.size < config.stableWindowSize) {
            return Pair("UNKNOWN", null)
        }

        // PASO 5: Majority vote - calcular moda
        val counts = predictionHistory.groupingBy { it.first }.eachCount()
        val maxCount = counts.maxByOrNull { it.value }?.value ?: 0
        val candidates = counts.filter { it.value == maxCount }.keys

        // Si hay empate (tie-break), RECHAZAR → mantener estado anterior
        if (candidates.size > 1) {
            return Pair(lastStableLabel ?: "UNKNOWN", lastStableIndex)
        }

        val candidate = candidates.first()

        // PASO 6: Regla especial para "stationary"
        // Solo aceptar "stationary" si cumple TODAS las condiciones:
        if (candidate == "stationary") {
            val stationaryEntries = predictionHistory.filter { it.first == "stationary" }
            val stationaryCount = stationaryEntries.size

            // (A) Mayoría estricta: count >= 3 (de 5)
            if (stationaryCount < config.minStationaryCount) {
                return Pair(lastStableLabel ?: "UNKNOWN", lastStableIndex)
            }

            // (B) Confianza promedio: mean(confidence) >= 0.65
            val meanStationaryConfidence = stationaryEntries
                .map { it.second }
                .average()
                .toFloat()

            if (meanStationaryConfidence < config.minConfidenceStationary) {
                return Pair(lastStableLabel ?: "UNKNOWN", lastStableIndex)
            }

            // (C) No es producto de tie-break (ya validado en PASO 5)
        }

        val stableIndex = config.classes.indexOf(candidate).takeIf { it >= 0 }
        return Pair(candidate, stableIndex)
    }
}
