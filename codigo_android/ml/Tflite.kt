package com.example.activitydetector.ml

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder


 //Gestor del modelo TensorFlow Lite para predicciones de actividad
 //Carga el modelo, el scaler y proporciona funcionalidad de predicción

class Tflite(context: Context) {

    private val interpreter: Interpreter
    private val scalerParams: ScalerParams

    init {
        // Cargar el modelo TFLite desde assets
        val modelBuffer = context.assets.loadModel("modelo.tflite")
        interpreter = Interpreter(modelBuffer)

        // Cargar los parámetros del scaler desde JSON
        scalerParams = loadScalerParams(context.assets)
    }

    /*
     * Realiza una predicción con los datos de entrada
     *
     * @ entra input array de dimensiones [1, 50, 4] (lote, secuencia, características)
     * @devuelve array de dimensiones [1, 3] con las probabilidades de cada actividad
     */
    fun predict(input: Array<Array<FloatArray>>): Array<FloatArray> {
        val output = Array(1) { FloatArray(3) }
        interpreter.run(input, output)
        return output
    }



     // Normaliza los datos de entrada usando los parámetros del scaler

    private fun normalizeInput(input: Array<Array<FloatArray>>): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * 50 * 4 * 4)
        buffer.order(ByteOrder.nativeOrder())

        for (i in input.indices) {
            for (j in input[i].indices) {
                for (k in input[i][j].indices) {
                    val normalized = (input[i][j][k] - scalerParams.mean[k]) / scalerParams.scale[k]
                    buffer.putFloat(normalized)
                }
            }
        }

        buffer.rewind()
        return buffer
    }


     //Carga los parámetros del scaler desde el archivo JSON

    private fun loadScalerParams(assetManager: AssetManager): ScalerParams {
        val jsonString = assetManager.open("scaler.json").bufferedReader().use {
            it.readText()
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
        val scale = FloatArray(4) { i -> stdArray.getDouble(i).toFloat() }

        return ScalerParams(mean, scale)
    }


     //Libera los recursos del intérprete

    fun close() {
        interpreter.close()
    }


     //Clase para almacenar los parámetros del scaler (media y escala)

    private data class ScalerParams(
        val mean: FloatArray,
        val scale: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as ScalerParams
            if (!mean.contentEquals(other.mean)) return false
            if (!scale.contentEquals(other.scale)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = mean.contentHashCode()
            result = 31 * result + scale.contentHashCode()
            return result
        }
    }
}


 // Extensión para cargar un archivo del AssetManager como ByteBuffer

private fun AssetManager.loadModel(fileName: String): ByteBuffer {
    val inputStream = this.open(fileName)
    val size = inputStream.available()
    val buffer = ByteArray(size)
    inputStream.read(buffer)
    inputStream.close()

    return ByteBuffer.allocateDirect(size).apply {
        order(ByteOrder.nativeOrder())
        put(buffer)
        rewind()
    }
}
