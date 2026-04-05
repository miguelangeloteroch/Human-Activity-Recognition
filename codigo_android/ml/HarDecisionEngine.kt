package com.example.activitydetector.ml


 //Contrato del motor de decision HAR
 // Recibir predicciones del modelo (probs) cada step=10 ticks
 //Consolidar decisiones mediante majority vote sobre ventana de N=5
 // Aplicar reglas especiales (stationary, histéresis)
 // Retornar decisión estable o UNKNOWN durante calentamiento

interface HarDecisionEngine {
//Definimos las funciones que van a existir

    fun reset()
    //resetea el estado interno del motor

    fun update(timestampMs: Long, probs: FloatArray): HarDecisionOutput

     // Procesa una nueva predicción del modelo.

     // @param timestampMs Timestamp en ms desde boot (event.timestamp / 1_000_000L)
     //@param probs Probabilidades del modelo [size=3], orden: [jogging, stationary, walking]
     // @return HarDecisionOutput con predicción cruda + decisión estable

}
