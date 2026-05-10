# Weighted_FedAvg.py

import numpy as np
import tensorflow as tf

from temp_feature import build_feature_matrix
from Sequence import create_user_sequences


# =====================================================
# 1️⃣ BUILD LSTM AUTOENCODER MODEL
# =====================================================

def build_model(time_steps, num_features):

    inputs = tf.keras.layers.Input(
        shape=(time_steps, num_features)
    )

    # =================================================
    # ENCODER
    # =================================================

    x = tf.keras.layers.LSTM(
        64,
        activation='relu'
    )(inputs)

    x = tf.keras.layers.Dense(
        32,
        activation='relu'
    )(x)

    # =================================================
    # REPEAT VECTOR
    # =================================================

    x = tf.keras.layers.RepeatVector(
        time_steps
    )(x)

    # =================================================
    # DECODER
    # =================================================

    x = tf.keras.layers.LSTM(
        64,
        activation='relu',
        return_sequences=True
    )(x)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_features)
    )(x)

    model = tf.keras.models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# =====================================================
# 2️⃣ WEIGHTED FEDAVG
# =====================================================

def weighted_fedavg(local_weights, client_sizes):

    total_samples = np.sum(client_sizes)

    new_weights = []

    for weights in zip(*local_weights):

        weighted_sum = np.zeros_like(weights[0])

        for w, n in zip(weights, client_sizes):

            weighted_sum += (
                (n / total_samples) * w
            )

        new_weights.append(weighted_sum)

    return new_weights


# =====================================================
# 3️⃣ LOAD DATA
# =====================================================

BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

print("Loading feature matrix...")

X_scaled, session_df = build_feature_matrix(
    BASE_PATH,
    nrows=100000
)

print("Creating sequences...")

window_size = 10

X_seq, sequence_users = create_user_sequences(
    X_scaled,
    session_df,
    window_size
)

print("Sequence Shape:", X_seq.shape)


# =====================================================
# 4️⃣ CREATE FEDERATED CLIENTS
# =====================================================

unique_users = np.unique(sequence_users)

num_clients = 3

user_groups = np.array_split(
    unique_users,
    num_clients
)

client_data = []
client_sizes = []

print("\nFederated Client Distribution:")

for i, group in enumerate(user_groups):

    mask = np.isin(
        sequence_users,
        group
    )

    client_sequences = X_seq[mask]

    client_data.append(client_sequences)

    client_sizes.append(
        len(client_sequences)
    )

    print(
        f"Client {i+1} "
        f"→ Users: {len(group)} "
        f"| Sequences: {len(client_sequences)}"
    )


# =====================================================
# 5️⃣ FEDERATED TRAINING
# =====================================================

time_steps = X_seq.shape[1]

num_features = X_seq.shape[2]

global_model = build_model(
    time_steps,
    num_features
)

rounds = 5

for r in range(rounds):

    print(f"\nFederated Round {r+1}")

    local_weights = []

    # ================================================
    # CLIENT TRAINING
    # ================================================

    for i, data in enumerate(client_data):

        print(f"Training Client {i+1}")

        local_model = build_model(
            time_steps,
            num_features
        )

        # initialize with global weights
        local_model.set_weights(
            global_model.get_weights()
        )

        local_model.fit(
            data,
            data,
            epochs=3,
            batch_size=128,
            verbose=1
        )

        local_weights.append(
            local_model.get_weights()
        )

    # ================================================
    # WEIGHTED AGGREGATION
    # ================================================

    print("\nPerforming Weighted Aggregation...")

    new_weights = weighted_fedavg(
        local_weights,
        client_sizes
    )

    global_model.set_weights(
        new_weights
    )

print("\nFederated Training Complete")


# =====================================================
# 6️⃣ ANOMALY DETECTION
# =====================================================

print("\nDetecting anomalies...")

X_pred = global_model.predict(
    X_seq,
    verbose=0
)

# reconstruction error
mse = np.mean(
    np.power(X_seq - X_pred, 2),
    axis=(1, 2)
)

# threshold
threshold = mse.mean() + 3 * mse.std()

print("Mean Error:", mse.mean())

print("Std Error:", mse.std())

print("Anomaly Threshold:", threshold)

# anomaly labels
anomalies = mse > threshold

print("\nTotal Sequences:", len(X_seq))

print("Detected Anomalies:", np.sum(anomalies))


# =====================================================
# 7️⃣ MAP ANOMALIES TO USERS
# =====================================================

anomalous_users = sequence_users[anomalies]

if len(anomalous_users) == 0:

    print("\nNo anomalous users detected.")

else:

    unique_users, counts = np.unique(
        anomalous_users,
        return_counts=True
    )

    print("\nUsers Responsible for Anomalies:")

    for user, count in zip(unique_users, counts):

        print(
            f"User: {user} "
            f"→ {count} anomalies"
        )

    # ================================================
    # TOP RISKY USERS
    # ================================================

    sorted_indices = np.argsort(-counts)

    print("\nTop Risky Users:")

    for idx in sorted_indices[:10]:

        print(
            f"User: {unique_users[idx]} "
            f"→ {counts[idx]} anomalies"
        )


# =====================================================
# 8️⃣ SAVE MODEL
# =====================================================

global_model.save(
    "weighted_fedavg_lstm_model.keras"
)

print(
    "\nWeighted FedAvg Global Model Saved"
)


# =====================================================
# 9️⃣ SAVE PREDICTIONS
# =====================================================

np.save(
    "weighted_fedavg_predictions.npy",
    anomalies
)

np.save(
    "weighted_fedavg_mse.npy",
    mse
)

print("Predictions saved successfully")