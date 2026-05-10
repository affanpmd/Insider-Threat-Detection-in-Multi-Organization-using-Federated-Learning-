# realtime_monitor.py

import time
from datetime import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from temp_feature import build_feature_matrix
from Sequence import create_user_sequences


# =====================================================
# 1️⃣ LOAD TRAINED MODEL
# =====================================================

print("Loading Weighted FedAvg model...")

model = tf.keras.models.load_model(
    "weighted_fedavg_lstm_model.keras"
)

print("Model loaded successfully")


# =====================================================
# 2️⃣ CONFIGURATION
# =====================================================

BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

window_size = 10

# initial dataset size
current_rows = 20000

# simulate incoming logs
rows_increment = 5000

# output file
csv_file = "realtime_results.csv"

print("\nStarting real-time monitoring...\n")


# =====================================================
# 3️⃣ REAL-TIME LOOP
# =====================================================

while True:

    print("\n===================================")
    print("Processing rows:", current_rows)
    print("===================================\n")

    # =================================================
    # LOAD FEATURE MATRIX
    # =================================================

    X_scaled, session_df = build_feature_matrix(
        BASE_PATH,
        current_rows
    )

    # =================================================
    # CREATE TEMPORAL SEQUENCES
    # =================================================

    X_seq, sequence_users = create_user_sequences(
        X_scaled,
        session_df,
        window_size
    )

    print("Sequence Shape:", X_seq.shape)

    # =================================================
    # RUN MODEL
    # =================================================

    print("Running anomaly detection...")

    X_pred = model.predict(
        X_seq,
        verbose=0
    )

    # =================================================
    # RECONSTRUCTION ERROR
    # =================================================

    mse = np.mean(
        np.power(X_seq - X_pred, 2),
        axis=(1, 2)
    )

    threshold = mse.mean() + 3 * mse.std()

    anomalies = mse > threshold

    # =================================================
    # BASIC STATS
    # =================================================

    total_sequences = len(X_seq)

    anomaly_count = np.sum(anomalies)

    anomaly_rate = (
        anomaly_count / total_sequences
    )

    anomalous_users = sequence_users[anomalies]

    # =================================================
    # DEFAULT VALUES
    # =================================================

    top_user = "None"

    suspicious_activity = (
        "No suspicious activity detected"
    )

    suspicious_time = "N/A"

    # =================================================
    # USER ANALYSIS
    # =================================================

    if len(anomalous_users) > 0:

        unique_users, counts = np.unique(
            anomalous_users,
            return_counts=True
        )

        # highest anomaly count user
        top_user = unique_users[
            np.argmax(counts)
        ]

        # =============================================
        # GET USER SESSION DATA
        # =============================================

        top_user_sessions = session_df[
            session_df["user"] == top_user
        ]

        # most recent session
        latest_session = top_user_sessions.iloc[-1]

        suspicious_time = str(
            latest_session["session_start"]
        )

        # =============================================
        # DETECT SUSPICIOUS ACTIVITIES
        # =============================================

        activity_flags = []

        # after-hours login
        if latest_session[
            "after_hours_login"
        ] == 1:

            activity_flags.append(
                "After-hours login"
            )

        # usb activity
        if latest_session[
            "usb_used"
        ] == 1:

            activity_flags.append(
                "USB device usage"
            )

        # excessive file copying
        if latest_session[
            "file_copy_count"
        ] > 20:

            activity_flags.append(
                "Excessive file copies"
            )

        # high web activity
        if latest_session[
            "web_count"
        ] > 50:

            activity_flags.append(
                "High web activity"
            )

        # suspicious emails
        if latest_session[
            "emails_sent"
        ] > 10:

            activity_flags.append(
                "Suspicious email activity"
            )

        # external communication
        if latest_session[
            "external_ratio"
        ] > 0.5:

            activity_flags.append(
                "External communication spike"
            )

        # abnormal session duration
        if latest_session[
            "session_duration"
        ] > 8:

            activity_flags.append(
                "Abnormally long session"
            )

        # weekend behavior
        if latest_session[
            "weekend_flag"
        ] == 1:

            activity_flags.append(
                "Weekend activity"
            )

        # =============================================
        # FINAL ACTIVITY STRING
        # =============================================

        if len(activity_flags) > 0:

            suspicious_activity = (
                " + ".join(activity_flags)
            )

        else:

            suspicious_activity = (
                "General anomalous behavior"
            )

    # =================================================
    # PRINT RESULTS
    # =================================================

    print("\n============= RESULTS =============")

    print("Timestamp:", datetime.now())

    print("Total Sequences:", total_sequences)

    print("Detected Anomalies:", anomaly_count)

    print("Anomaly Rate:", round(anomaly_rate, 5))

    print("Top Risky User:", top_user)

    print("Suspicious Activity:",
          suspicious_activity)

    print("Suspicious Session Time:",
          suspicious_time)

    print("===================================\n")

    # =================================================
    # SAVE TO CSV
    # =================================================

    new_row = pd.DataFrame([{

        "Timestamp":
            str(datetime.now()),

        "Rows Processed":
            current_rows,

        "Total Sequences":
            int(total_sequences),

        "Detected Anomalies":
            int(anomaly_count),

        "Anomaly Rate":
            float(round(anomaly_rate, 5)),

        "Top Risky User":
            str(top_user),

        "Suspicious Activity":
            suspicious_activity,

        "Suspicious Session Time":
            suspicious_time
    }])

    # =================================================
    # SAFE CSV SAVE
    # =================================================

    try:

        # create file
        if not os.path.exists(csv_file):

            new_row.to_csv(
                csv_file,
                index=False
            )

        # append
        else:

            new_row.to_csv(
                csv_file,
                mode='a',
                header=False,
                index=False
            )

        print("CSV updated successfully")

    except PermissionError:

        print(
            "\n⚠️ realtime_results.csv is open."
        )

        print(
            "Close the file to continue updates.\n"
        )

    # =================================================
    # SIMULATE NEW STREAMING LOGS
    # =================================================

    current_rows += rows_increment

    print(
        f"Next cycle will process "
        f"{current_rows} rows"
    )

    # =================================================
    # WAIT 15 SECONDS
    # =================================================

    print("\nWaiting 15 seconds...\n")

    time.sleep(15)