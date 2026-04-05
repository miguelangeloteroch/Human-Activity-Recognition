package com.example.activitydetector.ml

/**
 * Salida del motor de decisión HAR.
 *
 * Diferencia entre predicción cruda (raw) y decisión estable (stable):
 * - raw: predicción directa del modelo TFLite
 * - stable: decisión consolidada por majority vote (solo cuando historySize == N)
 *
 * Propiedades:
 * - timestampMs: Timestamp en ms desde boot
 * - probs: Probabilidades del modelo [jogging, stationary, walking]
 * - rawIndex/rawLabel/confidence: Predicción cruda (argmax)
 * - stableLabel/stableIndex/isStable: Decisión consolidada
 * - historySize/stableWindowSize: Diagnóstico de calentamiento
 */
data class HarDecisionOutput(
    val timestampMs: Long,

    // Predicción cruda (viene directa del modelo)
    val probs: FloatArray,
    val rawIndex: Int,
    val rawLabel: String,
    val confidence: Float,

    // Decisión estable (consolidada por majority vote)
    val stableLabel: String,
    val stableIndex: Int?,
    val isStable: Boolean,

    // Diagnóstico
    val historySize: Int,
    val stableWindowSize: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true //Si apuntan al mismo objeto en memorio son iguales
        if (javaClass != other?.javaClass) return false
        other as HarDecisionOutput

        if (timestampMs != other.timestampMs) return false
        if (!probs.contentEquals(other.probs)) return false
        if (rawIndex != other.rawIndex) return false
        if (rawLabel != other.rawLabel) return false
        if (confidence != other.confidence) return false
        if (stableLabel != other.stableLabel) return false
        if (stableIndex != other.stableIndex) return false
        if (isStable != other.isStable) return false
        if (historySize != other.historySize) return false
        if (stableWindowSize != other.stableWindowSize) return false

        return true //Si todo lo demás conincide se devuelve true
    }

    override fun hashCode(): Int {
        var result = timestampMs.hashCode()
        result = 31 * result + probs.contentHashCode()
        result = 31 * result + rawIndex
        result = 31 * result + rawLabel.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + stableLabel.hashCode()
        result = 31 * result + (stableIndex?.hashCode() ?: 0)
        result = 31 * result + isStable.hashCode()
        result = 31 * result + historySize
        result = 31 * result + stableWindowSize
        return result //Se genera un número entero que representa el contenido del objeto
    }
}
