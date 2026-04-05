package com.example.activitydetector.feature.auth.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.activitydetector.feature.auth.domain.AuthRepository
import com.example.activitydetector.feature.auth.domain.AuthResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    
    private val repo = AuthRepository(getApplication())
    
    private val _uiState = MutableStateFlow(AuthUiState())
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    init {
        // Al crear el ViewModel, intentamos recuperar el alias en memoria (si hay usuario activo).
        viewModelScope.launch {
            val alias = withContext(Dispatchers.IO) {
                repo.getCurrentUserAlias()
            }
            if (alias != null) {
                _uiState.update {
                    it.copy(
                        isLoggedIn = true,
                        // isLoggedInNow se usa solo para navegar tras login,
                        // así que no lo activamos aquí.
                        userAlias = alias
                    )
                }
            }
        }
    }
    
    //Limpiamos mensaje de error
    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }
    
    //Limpiamos mensaje de éxito
    fun clearSuccessMessage() {
        _uiState.update { it.copy(successMessage = null) }
    }
    
    /**
     * Iniciamos sesión con las credenciales proporcionadas.
     * 
     * @param userId ID del usuario
     * @param password Contraseña, la limpiamos al final
     */
    fun login(userId: String, password: CharArray) {
        viewModelScope.launch {
            try {
                _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
                
                val result = withContext(Dispatchers.IO) {
                    repo.login(userId, password)
                }
                
                when (result) {
                    is AuthResult.Success -> {
                        // Cargar alias del usuario
                        val alias = withContext(Dispatchers.IO) {
                            repo.getAliasForUserId(userId)

                        }
                        
                        // Marcar como logueado en memoria
                        _uiState.update { 
                            it.copy(
                                isLoading = false, 
                                isLoggedInNow = true,
                                isLoggedIn = true,
                                userAlias = alias
                            ) 
                        }
                    }
                    is AuthResult.Error -> {
                        _uiState.update { 
                            it.copy(isLoading = false, errorMessage = result.message) 
                        }
                    }
                }
            } finally {
                // Limpiar contraseña por seguridad
                password.fill('\u0000')
            }
        }
    }
    
    //REGISTRAMOS USUARIO NUEVO EN EL SISTEMA
    fun register(
        userId: String,
        password: CharArray,
        alias: String,
        age: Int,
        sex: String
    ) {
        viewModelScope.launch {
            try {
                _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
                
                val result = withContext(Dispatchers.IO) {
                    repo.register(userId, password, alias, age, sex)
                }
                
                when (result) {
                    is AuthResult.Success -> {
                        // NO marcar como logueado, solo mostrar mensaje de éxito
                        _uiState.update { 
                            it.copy(
                                isLoading = false,
                                successMessage = "Cuenta creada correctamente. Inicia sesión para continuar."
                            ) 
                        }
                    }
                    is AuthResult.Error -> {
                        _uiState.update { 
                            it.copy(isLoading = false, errorMessage = result.message) 
                        }
                    }
                }
            } finally {
                // Limpiar contraseña por seguridad
                password.fill('\u0000')
            }
        }
    }
    
    /**
     * Cierra la sesión del usuario activo.
     */
    fun logout() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                repo.logout()
            }
            _uiState.update { 
                it.copy(
                    isLoading = false, 
                    isLoggedInNow = false,
                    isLoggedIn = false
                ) 
            }
        }
    }
}

