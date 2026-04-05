package com.example.activitydetector.ui.listen

import android.app.Application
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Logout
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.activitydetector.feature.auth.ui.AuthViewModel

@Composable
fun ListenScreen(
    viewModel: ListenViewModel = viewModel(
        factory = androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory(
            LocalContext.current.applicationContext as Application
        )
    ),
    authViewModel: AuthViewModel,
    onActivityFinished: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val authState by authViewModel.uiState.collectAsStateWithLifecycle()
    
    // Actualizar callback cuando cambie el alias
    LaunchedEffect(authState.userAlias) {
        // Pasar el alias al ViewModel para el mensaje del countdown
        viewModel.setUserAlias(authState.userAlias)
        
        viewModel.onActivityFinished = {
            // Guardar el resultado final y el alias del usuario en estado compartido
            com.example.activitydetector.navigation.SharedActivityData.lastSessionResult = viewModel.lastSessionResult
            com.example.activitydetector.navigation.SharedActivityData.userAlias = authState.userAlias
            onActivityFinished()
        }
    }

    // Animación del color de fondo
    val backgroundColor by animateColorAsState(
        targetValue = Color(uiState.backgroundColor),
        animationSpec = tween(durationMillis = 350),
        label = "background_color"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundColor)
    ) {
        // Top-left: Mensaje de bienvenida y indicador
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(24.dp),
            horizontalAlignment = Alignment.Start,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            WelcomeMessage(
                userAlias = authState.userAlias
            )
            DetectingIndicator(
                isDetecting = uiState.isDetecting
            )
        }
        
        // Countdown de listening (encima del bloque central cuando está activo)
        if (uiState.currentActivity == ActivityState.UNKNOWN && uiState.isListeningCountdownActive) {
            ListeningCountdownMessage(
                userAlias = uiState.userAlias,
                secondsLeft = uiState.listeningSecondsLeft,
                modifier = Modifier
                    .align(Alignment.Center)
                    .offset(y = (-120).dp)
            )
        }

      //Historial de actividades (arriba derecha
        CompactHistoryList(
            segments = uiState.segmentsHistory,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(end = 24.dp, top = 24.dp) // Margen original,
                .offset(y = 40.dp) // desplazamiento hacia abajo

        )
        // BLOQUE CENTRAL
        CurrentActivityCenter(
            activity = uiState.currentActivity,
            elapsedMs = uiState.currentSegmentElapsedMs,
            activityText = uiState.activityDisplayText,
            formattedTime = uiState.formattedSegmentTime,
            modifier = Modifier.align(Alignment.Center)
        )
    }
}
//Mensaje de BIENVENIDA  a la sesión
@Composable
fun WelcomeMessage(
    userAlias: String?,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    
    LaunchedEffect(Unit) {
        visible = true
    }
    //Modificar el texto de bienvenido a la sesión
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(animationSpec = tween(280)) + 
                slideInVertically(
                    initialOffsetY = { -12 },
                    animationSpec = tween(280)
                ),
        modifier = modifier
    ) {
        val message = if (!userAlias.isNullOrBlank()) {
            "Bienvenido a la sesión, $userAlias"
        } else {
            "Bienvenido a la sesión"
        }
        
        Text(
            text = message,
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.3.sp,
            textAlign = TextAlign.Center
        )
    }
}

//Detectando actividad (el de debajo de bienvenida)

@Composable
fun DetectingIndicator(
    isDetecting: Boolean,
    modifier: Modifier = Modifier
) {
    Text(
        text = "● DETECTANDO ACTIVIDAD",
        color = Color.White.copy(alpha = 0.6f),
        fontSize = 12.sp,
        fontWeight = FontWeight.Normal,
        modifier = modifier
    )
}

@Composable
fun ListeningCountdownMessage(
    userAlias: String?,
    secondsLeft: Int,
    modifier: Modifier = Modifier
) {
    // Animación de parpadeo suave
    val infiniteTransition = rememberInfiniteTransition(label = "countdownBlink")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.5f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 800),
            repeatMode = RepeatMode.Reverse
        ),
        label = "countdownAlpha"
    )

    // Color dinámico según segundos restantes
    val baseColor = when {
        secondsLeft <= 5 -> Color(0xFFFF3B30)   // rojo chillón
        secondsLeft <= 10 -> Color(0xFFFF9500)  // naranja oscuro
        else -> Color.White
    }

    val animatedColor = baseColor.copy(alpha = alpha)

    Box(
        modifier = modifier.fillMaxSize(),          // ocupa TODA la pantalla
        contentAlignment = Alignment.Center         // centro exacto X + Y
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = if (!userAlias.isNullOrBlank()) {
                    "$userAlias, TE QUEDAN"
                } else {
                    "TE QUEDAN"
                },
                color = animatedColor,
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
                letterSpacing = 0.8.sp
            )

            Spacer(modifier = Modifier.height(14.dp))

            Text(
                text = "$secondsLeft SEGUNDOS",
                color = animatedColor,
                fontSize = 34.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                letterSpacing = 1.sp
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "PARA EMPEZAR LA ACTIVIDAD",
                color = animatedColor,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                textAlign = TextAlign.Center,
                letterSpacing = 0.5.sp
            )
        }
    }
}




@Composable
fun CurrentActivityCenter(
    activity: ActivityState,
    elapsedMs: Long,
    activityText: String,
    formattedTime: String,
    modifier: Modifier = Modifier
) {
    val isUnknown = activityText == "DETECTANDO"

    val infiniteTransition = rememberInfiniteTransition(label = "detectBlink")
    val blinkAlpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.25f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 650),
            repeatMode = RepeatMode.Reverse
        ),
        label = "blinkAlpha"
    )

    val contentAlpha = if (isUnknown) blinkAlpha else 1f

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Texto de actividad con animación Crossfade
        AnimatedContent(
            targetState = activityText,
            transitionSpec = {
                fadeIn(animationSpec = tween(300)) togetherWith
                        fadeOut(animationSpec = tween(300))
            },
            label = "activity_text"
        ) { text ->
            Text(
                text = text ,
                color = Color.White.copy(alpha = contentAlpha),
                fontSize = 32.sp,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // CRONÓMETRO DEL MEDIO, LO PRUEBO A BORRAR
        /*
        Text(
            text = formattedTime,
            color = Color.White.copy(alpha = contentAlpha),
            fontSize = 72.sp,
            fontWeight = FontWeight.ExtraBold,
            textAlign = TextAlign.Center,
            letterSpacing = 2.sp
        )
          */
    }
}
//DOMINIO ANIMABLE

@Composable
fun CompactHistoryList(
    segments: List<SegmentEntry>,
    modifier: Modifier = Modifier
) {
    if (segments.isEmpty()) {
        return
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.End,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        segments.take(6).forEach { segment ->
            HistoryItem(segment = segment)
        }
    }
}

@Composable
fun HistoryItem(segment: SegmentEntry) {
    val minutes = (segment.durationMs / 1000 / 60).toInt()
    val seconds = (segment.durationMs / 1000 % 60).toInt()
    val formattedTime = String.format("%02d:%02d", minutes, seconds)
    
    val activityLabel = when (segment.activity) {
        ActivityState.WALKING -> "Andando"
        ActivityState.JOGGING -> "Corriendo"
        ActivityState.STATIONARY -> "Quieto"
        ActivityState.UNKNOWN -> "Detectando"
    }

    Text(
        text = "$formattedTime  $activityLabel",
        color = Color.White.copy(alpha = 0.8f),
        fontSize = 13.sp,
        fontWeight = FontWeight.Normal,
        textAlign = TextAlign.End
    )
}
