# FEDAVG.py

import numpy as np
import tensorflow as tf

from temp_feature import build_feature_matrix
from Sequence import create_user_sequences


 # BUILD LSTM AUTOENCODER MODEL


def build_model(time_steps, num_features):

    inputs = tf.keras.layers.Input(shape=(time_steps, num_features))

    # Encoder
    x = tf.keras.layers.LSTM(64, activation='relu')(inputs)
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

    return model



# 2️⃣ FEDERATED AVERAGING FUNCTION


def federated_average(local_weights):

    avg_weights = []

    for weights in zip(*local_weights):
        avg_weights.append(np.mean(np.array(weights), axis=0))

    return avg_weights



# 3️⃣ LOAD DATA


BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

print("Loading feature matrix...")

X_scaled, session_df = build_feature_matrix(BASE_PATH)

print("Creating sequences...")

window_size = 10

X_seq, sequence_users = create_user_sequences(
    X_scaled, session_df, window_size
)

print("Sequence Shape:", X_seq.shape)



# 4️⃣ CREATE FEDERATED CLIENTS BY USER GROUPS


unique_users = np.unique(sequence_users)

num_clients = 3

user_groups = np.array_split(unique_users, num_clients)

client_data = []

print("\nFederated Client Distribution:")

for i, group in enumerate(user_groups):

    mask = np.isin(sequence_users, group)

    client_sequences = X_seq[mask]

    client_data.append(client_sequences)

    print(f"Client {i+1} → Users:", len(group),
          "| Sequences:", len(client_sequences))


# 5️⃣ FEDERATED TRAINING


time_steps = X_seq.shape[1]
num_features = X_seq.shape[2]

global_model = build_model(time_steps, num_features)

rounds = 5

for r in range(rounds):

    print(f"\nFederated Round {r+1}")

    local_weights = []

    for i, data in enumerate(client_data):

        print(f"Training Client {i+1}")

        local_model = build_model(time_steps, num_features)

        # start from global model weights
        local_model.set_weights(global_model.get_weights())

        local_model.fit(
            data,
            data,
            epochs=3,
            batch_size=128,
            verbose=0
        )

        local_weights.append(local_model.get_weights())

    # FedAvg aggregation
    new_weights = federated_average(local_weights)

    global_model.set_weights(new_weights)


print("\nFederated Training Complete")



# 6️⃣ ANOMALY DETECTION


print("\nDetecting anomalies...")

X_pred = global_model.predict(X_seq)

mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1,2))

threshold = mse.mean() + 3 * mse.std()

print("Anomaly Threshold:", threshold)

anomalies = mse > threshold

print("Total Sequences:", len(X_seq))
print("Detected Anomalies:", np.sum(anomalies))



# 7️⃣ MAP ANOMALIES TO USERS


anomalous_users = sequence_users[anomalies]

unique_users, counts = np.unique(anomalous_users, return_counts=True)

print("\nUsers Responsible for Anomalies:")

for user, count in zip(unique_users, counts):
    print(f"User: {user} → {count} anomalies")



# 8️⃣ TOP RISKY USERS
# =====================================================

sorted_indices = np.argsort(-counts)

print("\nTop Risky Users:")

for idx in sorted_indices[:10]:
    print(f"User: {unique_users[idx]} → {counts[idx]} anomalies")


# =====================================================
# 9️⃣ SAVE GLOBAL MODEL
# =====================================================

global_model.save("fedavg_lstm_model.keras")

print("\nFederated Global Model Saved")


#accuracy
np.save("fedavg_predictions.npy", anomalies)