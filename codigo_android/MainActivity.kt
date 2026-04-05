package com.example.activitydetector

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.compose.rememberNavController
import com.example.activitydetector.navigation.AppNavigation
import com.example.activitydetector.ui.theme.ActivityDetectorTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ActivityDetectorTheme {
                val navController = rememberNavController()
                AppNavigation(navController = navController)
            }
        }
    }
}
