package com.example.activitydetector.feature.auth.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.material3.ExperimentalMaterial3Api //Para usar dropdown

@OptIn(ExperimentalMaterial3Api::class) //Para usar dropdown
@Composable
fun AuthScreen(
    authViewModel: AuthViewModel,
    onContinue: () -> Unit
) {
    val viewModel = authViewModel

    val uiState by viewModel.uiState.collectAsState()

    // Modo: true = Login, false = Register
    var isLoginMode by remember { mutableStateOf(true) }

    // Campos comunes
    var userId by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    // Campos solo para registro
    var alias by remember { mutableStateOf("") }
    var age by remember { mutableStateOf("") }
    var sex by remember { mutableStateOf("") }
    var sexExpanded by remember { mutableStateOf(false) }
    val sexOptions = listOf("Hombre", "Mujer", "Otro")

    // Observar cuando se loguea exitosamente (solo login, no registro)
    LaunchedEffect(uiState.isLoggedInNow, isLoginMode) {
        if (isLoginMode && uiState.isLoggedInNow) {
            onContinue()
        }
    }


    // Observar mensaje de éxito tras registro
    LaunchedEffect(uiState.successMessage) {
        if (uiState.successMessage != null) {
            // Cambiar a modo login tras registro exitoso
            isLoginMode = true
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Card(
            modifier = Modifier
                .widthIn(max = 420.dp)
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = if (isLoginMode) "Iniciar sesión" else "Crear cuenta",
                    style = MaterialTheme.typography.headlineLarge,
                    color = MaterialTheme.colorScheme.onSurface
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = if (isLoginMode)
                        "Ingresa tus credenciales"
                    else
                        "Completa todos los campos para registrarte",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Campo: User ID
                OutlinedTextField(
                    value = userId,
                    onValueChange = {
                        userId = it
                        viewModel.clearError()
                        viewModel.clearSuccessMessage()
                    },
                    label = { Text("Usuario") },
                    leadingIcon = { Icon(Icons.Outlined.Person, contentDescription = "Usuario") },
                    singleLine = true,
                    enabled = !uiState.isLoading,
                    modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text)
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Campo: Password
                OutlinedTextField(
                    value = password,
                    onValueChange = {
                        password = it
                        viewModel.clearError()
                        viewModel.clearSuccessMessage()
                    },
                    label = { Text("Contraseña") },
                    leadingIcon = { Icon(Icons.Outlined.Lock, contentDescription = "Contraseña") },
                    singleLine = true,
                    enabled = !uiState.isLoading,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password)
                )

                // Ya no mostramos el switch de "Recordar sesión"; la sesión es solo en memoria.



                    // Campos adicionales solo para registro
                    if (!isLoginMode) {
                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = alias,
                            onValueChange = {
                                alias = it
                                viewModel.clearError()
                            },
                            label = { Text("Alias") },
                            singleLine = true,
                            enabled = !uiState.isLoading,
                            modifier = Modifier.fillMaxWidth(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text)
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = age,
                            onValueChange = {
                                if (it.all { char -> char.isDigit() }) {
                                    age = it
                                    viewModel.clearError()
                                }
                            },
                            label = { Text("Edad") },
                            singleLine = true,
                            enabled = !uiState.isLoading,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        // Dropdown de Sexo
                        ExposedDropdownMenuBox(
                            expanded = sexExpanded,
                            onExpandedChange = {
                                if (!uiState.isLoading) {
                                    sexExpanded = !sexExpanded
                                }
                            }
                        ) {
                            OutlinedTextField(
                                value = sex,
                                onValueChange = {},
                                readOnly = true,
                                label = { Text("Sexo") },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = sexExpanded) },
                                enabled = !uiState.isLoading,
                                modifier = Modifier
                                    .menuAnchor()
                                    .fillMaxWidth()
                            )
                            ExposedDropdownMenu(
                                expanded = sexExpanded,
                                onDismissRequest = { sexExpanded = false }
                            ) {
                                sexOptions.forEach { option ->
                                    DropdownMenuItem(
                                        text = { Text(option) },
                                        onClick = {
                                            sex = option
                                            sexExpanded = false
                                            viewModel.clearError()
                                        }
                                    )
                                }
                            }
                        }
                    }

                    // Mostrar mensaje de éxito si existe
                    if (uiState.successMessage != null) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = uiState.successMessage!!,
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    // Mostrar error si existe
                    if (uiState.errorMessage != null) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = uiState.errorMessage!!,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    Spacer(modifier = Modifier.height(32.dp))

                    // Botón principal (Login o Register)
                    Button(
                        onClick = {
                            if (isLoginMode) {
                                // Login
                                viewModel.login(
                                    userId = userId,
                                    password = password.toCharArray()
                                )
                            } else {
                                // Register
                                val ageInt = age.toIntOrNull() ?: 0
                                viewModel.register(
                                    userId = userId,
                                    password = password.toCharArray(),
                                    alias = alias,
                                    age = ageInt,
                                    sex = sex
                                )
                            }
                        },
                        enabled = !uiState.isLoading,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = MaterialTheme.shapes.medium
                    ) {
                        if (uiState.isLoading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(24.dp),
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                        } else {
                            Text(
                                text = if (isLoginMode) "Iniciar sesión" else "Registrarse",
                                style = MaterialTheme.typography.labelLarge
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Botón para cambiar de modo
                    TextButton(
                        onClick = {
                            isLoginMode = !isLoginMode
                            viewModel.clearError()
                            viewModel.clearSuccessMessage()
                        },
                        enabled = !uiState.isLoading
                    ) {
                        Text(
                            text = if (isLoginMode)
                                "¿No tienes cuenta? Regístrate"
                            else
                                "¿Ya tienes cuenta? Inicia sesión"
                        )
                    }
                }
            }
        }
    }

