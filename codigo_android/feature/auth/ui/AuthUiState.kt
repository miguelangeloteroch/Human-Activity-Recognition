package com.example.activitydetector.feature.auth.ui

data class AuthUiState(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val successMessage: String? = null,
    val isLoggedInNow: Boolean = false,
    val isLoggedIn: Boolean = false,  // Mantener para compatibilidad
    val userAlias: String? = null  // Alias del usuario
)

