package com.example.activitydetector.feature.auth.domain

import android.content.Context
import com.example.activitydetector.feature.auth.data.database.AppDatabase
import com.example.activitydetector.feature.auth.data.database.entity.UserEntity
import com.example.activitydetector.core.security.PasswordHasher

class AuthRepository(context: Context) {
    private val hasher: PasswordHasher = PasswordHasher

    private val appContext = context.applicationContext
    private val userDao = AppDatabase.getDatabase(appContext).userDao()

    companion object {
        // Estado de autenticación solo en memoria, compartido en todo el proceso
        private var currentUserId: String? = null

        fun setCurrentUserId(id: String?) {
            currentUserId = id
        }

        fun getCurrentUserId(): String? = currentUserId
    }
    
    /**
     * Registra un nuevo usuario en el sistema.
     * 
     * Validaciones:
     * - userId no vacío y sin espacios
     * - password length >= 8
     * - alias no vacío
     * - age > 0 y <= 120
     * - sex no vacío
     * @return AuthResult.Success si el registro fue exitoso, AuthResult.Error en caso contrario
     */
    suspend fun register(
        userId: String,
        password: CharArray,
        alias: String,
        age: Int,
        sex: String
    ): AuthResult {
        // Validar userId
        if (userId.isBlank()) {
            return AuthResult.Error("El ID de usuario no puede estar vacío")
        }
        if (userId.contains(" ")) {
            return AuthResult.Error("El ID de usuario no puede contener espacios")
        }
        
        // Validar password
        if (password.size < 8) {
            return AuthResult.Error("La contraseña debe tener al menos 8 caracteres")
        }
        
        // Validar alias
        if (alias.isBlank()) {
            return AuthResult.Error("El alias no puede estar vacío")
        }
        
        // Validar age
        if (age <= 0 || age > 120) {
            return AuthResult.Error("La edad debe estar entre 1 y 120")
        }
        
        // Validar sex
        if (sex.isBlank()) {
            return AuthResult.Error("El sexo no puede estar vacío")
        }
        
        // Verificar si el usuario ya existe
        if (userDao.existsUserId(userId)) {
            return AuthResult.Error("El usuario ya existe")
        }
        
        // Generar salt y hash de la contraseña
        val salt = hasher.generateSaltBase64()
        val passwordHash = hasher.hashPasswordBase64(password.copyOf(), salt)
        
        // Crear entidad de usuario
        val currentTime = System.currentTimeMillis()
        val user = UserEntity(
            userId = userId,
            passwordHash = passwordHash,
            passwordSalt = salt,
            alias = alias,
            age = age,
            sex = sex,
            createdAt = currentTime,
            updatedAt = currentTime
        )
        
        // Insertar en la base de datos
        return try {
            userDao.insertUser(user)
            
            AuthResult.Success
        } catch (e: Exception) {
            AuthResult.Error("Error al registrar el usuario: ${e.message}")
        }
    }
    
    /**
     * Inicia sesión con un usuario existente.
     * @return AuthResult.Success si las credenciales son correctas, AuthResult.Error en caso contrario
     */
    suspend fun login(userId: String, password: CharArray): AuthResult {
        // Obtener usuario de la base de datos
        val user = userDao.getUserById(userId)
            ?: return AuthResult.Error("Usuario o contraseña incorrectos")
        
        // Verificar contraseña
        val isValid = hasher.verify(
            password = password.copyOf(),
            saltBase64 = user.passwordSalt,
            expectedHashBase64 = user.passwordHash
        )
        
        return if (isValid) {
            // Autenticación correcta: guardamos usuario solo en memoria (compartido).
            setCurrentUserId(userId)
            AuthResult.Success
        } else {
            AuthResult.Error("Usuario o contraseña incorrectos")
        }
    }
    
    /**
     * Cierra la sesión del usuario activo.
     */
    suspend fun logout() {
        // Solo limpiar el estado en memoria compartido; sin DataStore.
        setCurrentUserId(null)
    }
    
    /**
     * Obtiene el alias del usuario actualmente autenticado en memoria.
     * @return El alias del usuario o null si no hay sesión en memoria
     */
    suspend fun getCurrentUserAlias(): String? {
        val userId = getCurrentUserId() ?: return null
        val user = userDao.getUserById(userId)
        return user?.alias
    }

    /**
     * Obtiene el alias de un usuario específico por su ID.
     * @param userId ID del usuario
     * @return El alias del usuario o null si no existe
     */
    suspend fun getAliasForUserId(userId: String): String? {
        val user = userDao.getUserById(userId)
        return user?.alias
    }
}

