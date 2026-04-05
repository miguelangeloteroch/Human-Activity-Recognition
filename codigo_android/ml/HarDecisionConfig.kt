package com.example.activitydetector.ml


 //Configuración centralizada para el motor de decisión HAR (Human Activity Recognition)

data class HarDecisionConfig(
    val classes: List<String> = listOf("jogging", "stationary", "walking"), //Etiquetas de actividad
    val stableWindowSize: Int = 5, //Tamaño de ventana para conssolidar una decisión
    val minConfidence: Float = 0.55f, //Umbral minimo de confianza para cambiar a jogging o walking
    val minConfidenceStationary: Float = 0.65f, //Umbral mas estricto para stationary
    val minStationaryCount: Int = 3, //Mayoria estricta para stationary
    val switchConfidence: Float = 0.60f, //Umbral para cambios rápidos
    val switchConsecutiveHits: Int = 2 //Con dos muestras seguidas con un umbral mayor del 60% cambiamos de waalking a jogging o viveversa
)