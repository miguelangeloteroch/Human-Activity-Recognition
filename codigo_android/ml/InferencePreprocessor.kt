package com.example.activitydetector.ml

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * Parámetros de normalización (StandardScaler) cargados desde scaler.json
 */
data class ScalerParams(
    val mean: FloatArray,    // [mean_x, mean_y, mean_z, mean_mag]
    val std: FloatArray      // [std_x, std_y, std_z, std_mag]
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ScalerParams
        if (!mean.contentEquals(other.mean)) return false
        if (!std.contentEquals(other.std)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = mean.contentHashCode()
        result = 31 * result + std.contentHashCode()
        return result
    }
}

/**
 * Ring buffer circular de tamaño fijo [windowSize x 4]
 * Almacena muestras normalizadas: [x, y, z, mag]
 * Orden temporal: índice 0 = más antiguo, índice (windowSize-1) = más reciente
 */
class CircularBuffer(private val windowSize: Int) {
    private val buffer = Array(windowSize) { FloatArray(4) }
    private var writeIndex = 0
    private var count = 0

    fun push(sample: FloatArray) {
        require(sample.size == 4) { "Sample debe tener exactamente 4 elementos" }
        buffer[writeIndex] = sample.copyOf()
        writeIndex = (writeIndex + 1) % windowSize
        if (count < windowSize) count++
    }

    fun isFull(): Boolean = count == windowSize

    fun toArray(): Array<FloatArray> {
        if (count < windowSize) return arrayOf()

        val result = Array(windowSize) { FloatArray(4) }
        for (i in 0 until windowSize) {
            val index = (writeIndex + i) % windowSize
            result[i] = buffer[index].copyOf()
        }
        return result
    }

    fun reset() {
        count = 0
        writeIndex = 0
    }
}

/**
 * Preprocesador para inferencia HAR en tiempo real
 * - Resamplea a 20 Hz (50 ms)
 * - Calcula magnitud y normaliza
 * - Mantiene buffer de 50 muestras (2.5 s)
 * - Last sample hold para buckets sin eventos
 */
class InferencePreprocessorWithNormalization(context: Context) {

    companion object {
        private const val RESAMPLE_PERIOD_MS = 50L
        private const val WINDOW_SIZE = 50
    }

    private val scalerParams: ScalerParams
    private val buffer = CircularBuffer(WINDOW_SIZE)

    private var lastEventTimestampMs: Long = 0
    private var currentBucketStart: Long = 0
    private var currentBucketAccumulator = AccumulatorSampleWithNorm()
    private var lastResampledSample = FloatArray(4)
    private var hasLastResampled = false

    init {
        scalerParams = loadScalerParams(context.assets)
    }

    //Esta función procesa eventos del acelerómetro y resamplea a 20Hz.
    //Devuelve el número de muestras resampleadas quese pushean al buffer en esta llamda
    fun processAccelerometerEvent(timestampMs: Long, x: Float, y: Float, z: Float): Int {
        var pushed = 0  // contador de muestras pusheadas

        if (lastEventTimestampMs == 0L) {
            lastEventTimestampMs = timestampMs
            currentBucketStart = (timestampMs / RESAMPLE_PERIOD_MS) * RESAMPLE_PERIOD_MS
            currentBucketAccumulator = AccumulatorSampleWithNorm(x, y, z)
            return 0
        }

        val eventBucketIndex = timestampMs / RESAMPLE_PERIOD_MS
        val currentBucketIndex = currentBucketStart / RESAMPLE_PERIOD_MS

        if (eventBucketIndex == currentBucketIndex) {
            currentBucketAccumulator.add(x, y, z)
            lastEventTimestampMs = timestampMs
            return 0
        }

        // Evento en bucket diferente: cerrar buckets hasta llegar al bucket del evento
        while (currentBucketStart + RESAMPLE_PERIOD_MS <= timestampMs) {
            // Generar 1 muestra por iteración: real si hay eventos, hold si está vacío
            val sampleToPush = if (currentBucketAccumulator.hasData()) {
                currentBucketAccumulator.average(scalerParams)
            } else {
                if (hasLastResampled) lastResampledSample.copyOf()
                else {
                    currentBucketStart += RESAMPLE_PERIOD_MS
                    currentBucketAccumulator = AccumulatorSampleWithNorm()
                    continue
                }
            }
            lastResampledSample = sampleToPush
            buffer.push(sampleToPush)
            pushed++  // incrementar contador

            // Avanzar al siguiente bucket
            currentBucketStart += RESAMPLE_PERIOD_MS

            // Reiniciar acumulador vacío
            currentBucketAccumulator = AccumulatorSampleWithNorm()
        }

        // Preparar acumulador para el bucket donde cae el evento actual
        currentBucketAccumulator = AccumulatorSampleWithNorm(x, y, z)
        lastEventTimestampMs = timestampMs

        return pushed
    }

    fun hasWindow(): Boolean = buffer.isFull()

    fun getWindow(): Array<FloatArray> = buffer.toArray()

    // Reiniciar preprocesador
    fun reset() {
        buffer.reset()
        currentBucketAccumulator = AccumulatorSampleWithNorm()
        lastResampledSample = FloatArray(4)
        lastEventTimestampMs = 0
        currentBucketStart = 0
    }

    private fun loadScalerParams(assetManager: AssetManager): ScalerParams {
        val jsonString = try {
            assetManager.open("scaler.json").bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            throw RuntimeException("No se pudo cargar scaler.json desde assets", e)
        }

        val jsonObject = JSONObject(jsonString)

        // Compatibilidad: buscar en scaler_info (estructura extendida) o raíz
        val scalerInfo = if (jsonObject.has("scaler_info")) {
            jsonObject.getJSONObject("scaler_info")
        } else {
            jsonObject
        }

        val meanArray = scalerInfo.getJSONArray("mean")
        val stdArray = scalerInfo.getJSONArray("std")

        require(meanArray.length() == 4) { "mean debe tener 4 elementos" }
        require(stdArray.length() == 4) { "std debe tener 4 elementos" }

        val mean = FloatArray(4) { i -> meanArray.getDouble(i).toFloat() }
        val std = FloatArray(4) { i -> stdArray.getDouble(i).toFloat() }

        return ScalerParams(mean, std)
    }

    private class AccumulatorSampleWithNorm(
        private var sumX: Float = 0f,
        private var sumY: Float = 0f,
        private var sumZ: Float = 0f,
        private var count: Int = 0
    ) {
        constructor(x: Float, y: Float, z: Float) : this(x, y, z, 1)

        fun add(x: Float, y: Float, z: Float) {
            sumX += x
            sumY += y
            sumZ += z
            count++
        }

        fun hasData(): Boolean = count > 0

        fun average(scalerParams: ScalerParams): FloatArray {
            val avgX = if (count > 0) sumX / count else 0f
            val avgY = if (count > 0) sumY / count else 0f
            val avgZ = if (count > 0) sumZ / count else 0f
            val mag = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ)

            return floatArrayOf(
                (avgX - scalerParams.mean[0]) / scalerParams.std[0],
                (avgY - scalerParams.mean[1]) / scalerParams.std[1],
                (avgZ - scalerParams.mean[2]) / scalerParams.std[2],
                (mag - scalerParams.mean[3]) / scalerParams.std[3]
            )
        }
    }
}
