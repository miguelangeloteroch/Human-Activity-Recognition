package com.example.activitydetector.feature.auth.data.preferences

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_preferences")

class UserPreferences(private val context: Context) {
    companion object {
        private val USER_NAME_KEY = stringPreferencesKey("user_name")
        private val USER_AGE_KEY = intPreferencesKey("user_age")
    }

    val userFlow: Flow<User?> = context.dataStore.data.map { preferences ->
        val name = preferences[USER_NAME_KEY]
        val age = preferences[USER_AGE_KEY]
        if (name != null && age != null) {
            User(name, age)
        } else {
            null
        }
    }

   
}
data class User(
    val name: String,
    val age: Int
)

