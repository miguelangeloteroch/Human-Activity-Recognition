package com.example.activitydetector.ui.summary

import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.activitydetector.ui.listen.ActivityState
import com.example.activitydetector.ui.listen.SegmentEntry
import java.util.Locale

@Composable
fun SummaryScreen(
    viewModel: SummaryViewModel = viewModel(),
    onRetryActivity: () -> Unit = {}
) {
    // Cargar el resultado de la sesión y el alias del usuario
    val userAlias = remember { com.example.activitydetector.navigation.SharedActivityData.userAlias }
    
    LaunchedEffect(Unit) {
        val sessionResult = com.example.activitydetector.navigation.SharedActivityData.lastSessionResult
        if (sessionResult != null) {
            viewModel.setSessionResult(sessionResult)
        }
    }

    val state by viewModel.summaryState.collectAsState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                color = Color(0xFF1F2937) // Fondo oscuro profesional
            )
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
            contentPadding = PaddingValues(vertical = 24.dp)
        ) {
            // Encabezado
            item {
                SummaryHeader(userAlias = userAlias)
            }

            // Tarjeta de resumen superior (tiempo total)
            item {
                AnimatedVisibility(
                    visible = true,
                    enter = slideInVertically(
                        initialOffsetY = { -it },
                        animationSpec = tween(600)
                    ) + fadeIn(animationSpec = tween(600)),
                    exit = slideOutVertically() + fadeOut(),
                    label = "SummaryStatsCard"
                ) {
                    SummaryStatsCard(
                        state = state,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }

            // Desglose de tiempos (WALKING / JOGGING)
            item {
                TimeBreakdownCards(state = state)
            }

            // Historial de segmentos
            if (state.segments.isNotEmpty()) {
                item {
                    Text(
                        text = "Desgranado de actividad",
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                items(state.segments, key = { it.endedAtEpochMs }) { segment ->
                    AnimatedVisibility(
                        visible = true,
                        enter = slideInHorizontally(
                            initialOffsetX = { it / 2 },
                            animationSpec = tween(400)
                        ) + fadeIn(animationSpec = tween(400)),
                        exit = slideOutHorizontally() + fadeOut(),
                        label = "SegmentItem"
                    ) {
                        SegmentListItem(
                            segment = segment,
                            getActivityLabel = state::getActivityLabel,
                            getActivityColor = state::getActivityColor,
                            formatTime = state::formatTime,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }

            // Espaciador antes del botón
            item {
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Botón de reintentar
            item {
                AnimatedVisibility(
                    visible = true,
                    enter = slideInVertically(
                        initialOffsetY = { it },
                        animationSpec = tween(600)
                    ) + fadeIn(animationSpec = tween(600)),
                    exit = slideOutVertically() + fadeOut(),
                    label = "RetryButton"
                ) {
                    RetryButton(
                        onRetry = onRetryActivity,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }

            // Padding inferior
            item {
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
fun SummaryHeader(
    userAlias: String?,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    
    LaunchedEffect(Unit) {
        visible = true
    }
    
    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        
        Spacer(modifier = Modifier.height(12.dp))
        
        // Separador sutil
        HorizontalDivider(
            modifier = Modifier.fillMaxWidth(0.3f),
            thickness = 1.dp,
            color = Color.White.copy(alpha = 0.25f)
        )
        
        Spacer(modifier = Modifier.height(12.dp))
        
        Text(
            text = "Actividad Completada",
            color = Color.White,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Aquí tienes el resumen de tu sesión, $userAlias",
            color = Color.White.copy(alpha = 0.6f),
            fontSize = 14.sp,
            fontWeight = FontWeight.Normal,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
fun SummaryStatsCard(
    state: SummaryUiState,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp)),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF2D3748)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Tiempo Total de Actividad",
                color = Color.White.copy(alpha = 0.7f),
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 0.5.sp
            )

            Text(
                text = state.formatTime(state.totalActivityMs),
                color = Color.White,
                fontSize = 56.sp,
                fontWeight = FontWeight.ExtraBold,
                letterSpacing = 2.sp
            )

            Divider(
                modifier = Modifier
                    .fillMaxWidth(0.4f)
                    .height(1.dp),
                color = Color.White.copy(alpha = 0.2f)
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                StatBox(
                    label = "Andando",
                    value = state.formatTime(state.walkingMs),
                    color = Color(0xFFFF9F1C),
                    modifier = Modifier.weight(1f)
                )
                StatBox(
                    label = "Corriendo",
                    value = state.formatTime(state.joggingMs),
                    color = Color(0xFFE63946),
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
fun StatBox(
    label: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = value,
            color = color,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            color = Color.White.copy(alpha = 0.5f),
            fontSize = 11.sp,
            fontWeight = FontWeight.Normal
        )
    }
}

@Composable
fun TimeBreakdownCards(
    state: SummaryUiState,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "Desglose por Actividad",
            color = Color.White.copy(alpha = 0.7f),
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold
        )

        // Tarjeta Andando
        BreakdownCard(
            activity = ActivityState.WALKING,
            label = "Andando",
            time = state.formatTime(state.walkingMs),
            color = Color(0xFFFF9F1C),
            percentage = state.getWalkingPercentage()  // SOURCE OF TRUTH
        )

        // Tarjeta corriendo
        BreakdownCard(
            activity = ActivityState.JOGGING,
            label = "Corriendo",
            time = state.formatTime(state.joggingMs),
            color = Color(0xFFE63946),
            percentage = state.getJoggingPercentage()  // 100 - walking%
        )
    }
}

@Composable
fun BreakdownCard(
    activity: ActivityState,
    label: String,
    time: String,
    color: Color,
    percentage: Int,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp)),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF3D4758)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = label,
                    color = color,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = time,
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(60.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(color.copy(alpha = 0.2f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "$percentage%",
                        color = color,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
fun SegmentListItem(
    segment: SegmentEntry,
    getActivityLabel: (ActivityState) -> String,
    getActivityColor: (ActivityState) -> Long,
    formatTime: (Long) -> String,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp)),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF374151)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Badge de actividad
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(Color(getActivityColor(segment.activity)))
            )

            Spacer(modifier = Modifier.width(12.dp))

            // Label y tiempo
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                Text(
                    text = getActivityLabel(segment.activity),
                    color = Color.White,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = formatTime(segment.durationMs),
                    color = Color.White.copy(alpha = 0.6f),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Normal
                )
            }
        }
    }
}

@Composable
fun RetryButton(
    onRetry: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onRetry,
        modifier = modifier
            .height(56.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color(0xFFFF9F1C),
            contentColor = Color.White
        ),
        shape = RoundedCornerShape(12.dp),
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 6.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Filled.PlayArrow,
                contentDescription = "Reintentar",
                tint = Color.White
            )
            Text(
                text = "Grabar Nueva Actividad",
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 0.5.sp
            )
        }
    }
}
