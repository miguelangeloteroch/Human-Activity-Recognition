package com.example.activitydetector.feature.auth.domain

sealed class AuthResult {
    object Success : AuthResult()
    data class Error(val message: String) : AuthResult()
}

