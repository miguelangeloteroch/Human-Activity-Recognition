package com.example.activitydetector.navigation

import android.app.Application
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.activitydetector.feature.auth.ui.AuthScreen
import com.example.activitydetector.feature.auth.ui.AuthViewModel
import com.example.activitydetector.feature.auth.ui.AuthViewModelFactory
import com.example.activitydetector.ui.listen.ListenScreen
import com.example.activitydetector.ui.summary.SummaryScreen


//TABLA DE RUTAS
sealed class Screen(val route: String) {
    object Auth : Screen("auth")
    object Listen : Screen("listen")
    object Summary : Screen("summary")
}

// Estado compartido
//Con esto compartimos datos entre pantallas
object SharedActivityData {
    var lastSessionResult: com.example.activitydetector.data.SessionResult? = null
    var userAlias: String? = null
}

@Composable
fun AppNavigation(
    navController: NavHostController,
    startDestination: String = Screen.Auth.route //Empezamos en Auth
) {
    // Crear AuthViewModel una sola vez, aquí en AppNavigation
    // Se pasa como parámetro a AuthScreen y ListenScreen
    val application = LocalContext.current.applicationContext as Application
    val authViewModel: AuthViewModel = viewModel(
        factory = AuthViewModelFactory(application)
    )

    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        composable(Screen.Auth.route) { //Flujo AUTH LISTEN
            AuthScreen(
                authViewModel = authViewModel,
                onContinue = {
                    navController.navigate(Screen.Listen.route) {
                        popUpTo(Screen.Auth.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.Listen.route) {

            ListenScreen(
                authViewModel = authViewModel,
                onActivityFinished = { //De Escuchar a Pantalla Resumen
                    navController.navigate(Screen.Summary.route) {
                        popUpTo(Screen.Listen.route) { inclusive = false }
                    }
                }
            )
        }


        composable(Screen.Summary.route) { //De pantalla resumen a Esuchar de nuevo
            SummaryScreen(
                onRetryActivity = {
                    navController.navigate(Screen.Listen.route) {
                        popUpTo(Screen.Summary.route) { inclusive = true }
                    }
                }
            )
        }
    }
}




