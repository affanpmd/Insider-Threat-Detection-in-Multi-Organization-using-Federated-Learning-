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

# 🔥 FIXED: unpack both outputs
X_seq, sequence_users = create_user_sequences(
    X_scaled, session_df, window_size
)

print("Sequence Shape:", X_seq.shape)


# =====================================================
# 2️⃣ BUILD LSTM AUTOENCODER
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

model.compile(optimizer='adam', loss='mse')

model.summary()


# =====================================================
# 3️⃣ TRAIN MODEL
# =====================================================

print("\nTraining model...")

model.fit(
    X_seq,
    X_seq,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)


# =====================================================
# 4️⃣ CALCULATE RECONSTRUCTION ERROR
# =====================================================

print("\nCalculating reconstruction errors...")

X_pred = model.predict(X_seq)

mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))

print("Mean Error:", mse.mean())
print("Std Error:", mse.std())

threshold = mse.mean() + 3 * mse.std()

print("Anomaly Threshold:", threshold)


# =====================================================
# 5️⃣ DETECT ANOMALIES
# =====================================================

anomalies = mse > threshold

print("Total Sequences:", len(mse))
print("Detected Anomalies:", np.sum(anomalies))

print("First 20 anomaly indices:", np.where(anomalies)[0][:20])


# =====================================================
# 6️⃣ MAP ANOMALIES TO USERS
# =====================================================

anomalous_users = sequence_users[anomalies]

if len(anomalous_users) == 0:
    print("\nNo anomalous users detected.")
else:
    unique_users, counts = np.unique(anomalous_users, return_counts=True)

    print("\nUsers Responsible for Anomalies:")
    for user, count in zip(unique_users, counts):
        print(f"User: {user} → {count} anomalous sequences")

    # Sort users by highest anomaly count
    sorted_indices = np.argsort(-counts)

    print("\nTop Risky Users:")
    for idx in sorted_indices[:10]:
        print(f"User: {unique_users[idx]} → {counts[idx]} anomalies")
# =====================================================
# 7️⃣ USER RISK SCORING
# =====================================================

print("\nCalculating user risk scores...")

user_risk = {}
user_counts = {}

for i, user in enumerate(sequence_users):

    score = mse[i]

    if user not in user_risk:
        user_risk[user] = 0
        user_counts[user] = 0
        

    user_risk[user] += score
    user_counts[user] += 1


# Normalize risk by number of sequences
for user in user_risk:
    user_risk[user] = user_risk[user] / user_counts[user]


# Convert to sorted list
sorted_users = sorted(user_risk.items(), key=lambda x: x[1], reverse=True)


print("\nTop 10 Highest Risk Users:\n")

for user, score in sorted_users[:10]:
    print(f"User: {user} | Risk Score: {score:.6f}")

# =====================================================
# 7️⃣ OPTIONAL: SAVE MODEL
# =====================================================

model.save("lstm_autoencoder_model.h5")
print("\nModel saved as lstm_autoencoder_model.h5")

# Acuracy 
np.save("lstm_predictions.npy", anomalies)
