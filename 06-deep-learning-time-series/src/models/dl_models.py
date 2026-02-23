"""
Deep Learning Models for Time Series Forecasting
===================================================

Implements LSTM, GRU, and Transformer architectures using TensorFlow/Keras.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_shape: tuple, units: int = 64,
                     dropout: float = 0.2) -> keras.Model:
    """
    LSTM model with two stacked layers and dropout regularization.

    Architecture:
        Input -> LSTM(units, return_sequences) -> Dropout ->
        LSTM(units//2) -> Dropout -> Dense(32) -> Dense(1)
    """
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ], name="LSTM_Model")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def build_gru_model(input_shape: tuple, units: int = 64,
                    dropout: float = 0.2) -> keras.Model:
    """
    GRU model: fewer parameters than LSTM, often comparable performance.
    """
    model = keras.Sequential([
        layers.GRU(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.GRU(units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ], name="GRU_Model")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


class TransformerBlock(layers.Layer):
    """Single Transformer encoder block with multi-head self-attention."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=n_heads,
                                              key_dim=d_model // n_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)


def build_transformer_model(input_shape: tuple, d_model: int = 64,
                             n_heads: int = 4, ff_dim: int = 128,
                             n_blocks: int = 2, dropout: float = 0.1
                             ) -> keras.Model:
    """
    Transformer encoder for time series.
    Input projection -> Positional encoding -> N x TransformerBlock ->
    GlobalAveragePooling -> Dense -> Output
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)

    # Learnable positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_emb = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)
    x = x + pos_emb

    for _ in range(n_blocks):
        x = TransformerBlock(d_model, n_heads, ff_dim, dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="Transformer_Model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model
