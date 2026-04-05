package com.example.activitydetector.sensor

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

// Controla el acelerómetro y expone sus datos como un Flow
class SensorController(
    private val sensorManager: SensorManager
) {

    // Sensor y listener actuales
    private var sensor: Sensor? = null
    private var listener: SensorEventListener? = null

    // Al crear la clase, obtiene el acelerómetro por defecto
    init {
        sensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    // Devuelve un Flow con las lecturas [x, y, z] del acelerómetro
    fun accelerometerFlow(): Flow<FloatArray> = callbackFlow {

        // Obtener el acelerómetro
        val accelerometerSensor =
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        // Si no existe el sensor, cerrar el flow
        if (accelerometerSensor == null) {
            close()
            return@callbackFlow
        }

        // Listener que recibe los datos del sensor
        val eventListener = object : SensorEventListener {

            // Se ejecuta cada vez que hay nuevos datos
            override fun onSensorChanged(event: SensorEvent) {
                if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {

                    // Guardar valores X, Y, Z
                    val values = floatArrayOf(
                        event.values[0],
                        event.values[1],
                        event.values[2]
                    )

                    // Enviar datos al Flow
                    trySend(values)
                }
            }

            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
                // No se usa
            }
        }

        listener = eventListener

        try {
            // Registrar el listener (~60 Hz)
            sensorManager.registerListener(
                eventListener,
                accelerometerSensor,
                SensorManager.SENSOR_DELAY_UI
            )
            //EXCEPCIONES
        } catch (e: SecurityException) {
            // Por falta de permisos
            close(e)
            return@callbackFlow
        } catch (e: Exception) {
            // Por otro error
            close(e)
            return@callbackFlow
        }

        // Cuando se deja de observar el Flow, se elimina el listener
        awaitClose {
            listener?.let { sensorManager.unregisterListener(it) }
            listener = null
        }
    }

    // Detener el sensor manualmente
    fun stop() {
        listener?.let { sensorManager.unregisterListener(it) }
        listener = null
    }
}



