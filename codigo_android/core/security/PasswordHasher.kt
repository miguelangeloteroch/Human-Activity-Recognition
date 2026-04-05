package com.example.activitydetector.core.security

import android.util.Base64
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec
import java.security.MessageDigest

object PasswordHasher {

    private const val KEY_LENGTH_BITS = 256
    private const val SALT_LENGTH_BYTES = 16
    private const val ALGORITHM = "PBKDF2WithHmacSHA256"
    private const val ITERATIONS = 100_000

//Se genera un salt aleatorio
    fun generateSaltBase64(): String {
    val salt = ByteArray(SALT_LENGTH_BYTES)
    SecureRandom().nextBytes(salt)
    return Base64.encodeToString(salt, Base64.NO_WRAP)
}
//Hasheamos una contraseña usando PBKDF2
    //devolvemos el HASH de la contraseña en base64

    fun hashPasswordBase64(password: CharArray, saltBase64: String): String {
        return try {
            val salt = Base64.decode(saltBase64, Base64.NO_WRAP)
            val spec = PBEKeySpec(password, salt, ITERATIONS, KEY_LENGTH_BITS)
            val factory = SecretKeyFactory.getInstance(ALGORITHM)
            val hash = factory.generateSecret(spec).encoded
            Base64.encodeToString(hash, Base64.NO_WRAP)
        } finally {
            // Limpiamos chararray por seguridad
            password.fill('\u0000')
        }
    }
//Verificacimos si contraseña coincide con HASH

    fun verify(
        password: CharArray, //contraseña a verificar
        saltBase64: String, //salt usado en hash original
        expectedHashBase64: String //hash esperado
    ): Boolean {
        val passwordCopy = password.copyOf()
        val computedHash = hashPasswordBase64(passwordCopy, saltBase64)
        return MessageDigest.isEqual(
            Base64.decode(computedHash, Base64.NO_WRAP),
            Base64.decode(expectedHashBase64, Base64.NO_WRAP)
        )
    }
}

