package com.example.activitydetector.feature.auth.data.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey
    val userId: String,
    val passwordHash: String,
    val passwordSalt: String,
    val alias: String,
    val age: Int,
    val sex: String,
    val createdAt: Long,
    val updatedAt: Long
)

