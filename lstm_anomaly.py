# lstm_anomaly.py

import numpy as np
import tensorflow as tf
from temp_feature import build_feature_matrix
from Sequence import create_user_sequences


# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

print("Loading feature matrix...")
X_scaled, session_df = build_feature_matrix(BASE_PATH)

print("Creating sequences...")
window_size = 10
X_seq = create_user_sequences(X_scaled, session_df, window_size)

print("Sequence Shape:", X_seq.shape)


# =====================================================
# 2️⃣ BUILD MODEL
# =====================================================

time_steps = X_seq.shape[1]
num_features = X_seq.shape[2]

inputs = tf.keras.layers.Input(shape=(time_steps, num_features))

# Encoder
x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=False)(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)

# Repeat
x = tf.keras.layers.RepeatVector(time_steps)(x)

# Decoder
x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(num_features)
)(x)

model = tf.keras.models.Model(inputs, outputs)

# 👇 NO optimizer import needed
model.compile(optimizer='adam', loss='mse')

model.summary()


# =====================================================
# 3️⃣ TRAIN
# =====================================================

print("\nTraining model...")

model.fit(
    X_seq,
    X_seq,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    shuffle=True
)


# =====================================================
# 4️⃣ RECONSTRUCTION ERROR
# =====================================================

print("\nCalculating reconstruction errors...")

X_pred = model.predict(X_seq)

mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))

print("Mean Error:", mse.mean())
print("Std Error:", mse.std())

threshold = mse.mean() + 3 * mse.std()

print("Anomaly Threshold:", threshold)

anomalies = mse > threshold

print("Total Sequences:", len(mse))
print("Detected Anomalies:", np.sum(anomalies))

print("First 20 anomaly indices:", np.where(anomalies)[0][:20])