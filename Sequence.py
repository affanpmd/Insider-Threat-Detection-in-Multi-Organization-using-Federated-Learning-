# sequence.py

import numpy as np
from temp_feature import build_feature_matrix


def create_user_sequences(X_scaled, session_df, window_size=10):
    """
    Converts session features into user-wise temporal sequences.

    Returns:
        sequences : numpy array (samples, time_steps, features)
    """

    sequences = []

    feature_cols = session_df.columns.drop("user")
    session_df_scaled = session_df.copy()
    session_df_scaled[feature_cols] = X_scaled

    for user, user_data in session_df_scaled.groupby("user"):

        user_data = user_data.reset_index(drop=True)
        user_features = user_data[feature_cols].values

        if len(user_features) < window_size:
            continue

        for i in range(len(user_features) - window_size + 1):
            seq = user_features[i:i + window_size]
            sequences.append(seq)

    return np.array(sequences)


if __name__ == "__main__":

    BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

    X_scaled, session_df = build_feature_matrix(BASE_PATH)

    window_size = 10
    X_seq = create_user_sequences(X_scaled, session_df, window_size)

    print("Sequence Shape:", X_seq.shape)
    print("Single Sequence Shape:", X_seq[0].shape)