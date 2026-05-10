# sequence.py

import numpy as np


# =====================================================
# CREATE USER SEQUENCES
# =====================================================

def create_user_sequences(
    X_scaled,
    session_df,
    window_size=10
):

    """
    Converts session features into user-wise
    temporal sequences.

    Returns:
        sequences       :
            numpy array
            (samples, time_steps, features)

        sequence_users  :
            user mapped to each sequence
    """

    sequences = []
    sequence_users = []

    # =================================================
    # REMOVE NON-FEATURE COLUMNS
    # =================================================

    feature_cols = session_df.columns.drop([

        "user",

        # timestamps should NOT go into LSTM
        "session_start",
        "session_end"
    ])

    # =================================================
    # CREATE SCALED DATAFRAME
    # =================================================

    session_df_scaled = session_df.copy()

    session_df_scaled[feature_cols] = X_scaled

    # =================================================
    # USER-WISE SEQUENCE CREATION
    # =================================================

    for user, user_data in session_df_scaled.groupby("user"):

        # sort by session time
        user_data = user_data.sort_values(
            "session_start"
        ).reset_index(drop=True)

        # only numerical features
        user_features = user_data[
            feature_cols
        ].values

        # skip users with few sessions
        if len(user_features) < window_size:
            continue

        # =============================================
        # SLIDING WINDOW
        # =============================================

        for i in range(
            len(user_features) - window_size + 1
        ):

            seq = user_features[
                i:i + window_size
            ]

            sequences.append(seq)

            sequence_users.append(user)

    # =================================================
    # CONVERT TO NUMPY
    # =================================================

    sequences = np.array(sequences)

    sequence_users = np.array(sequence_users)

    return sequences, sequence_users


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    from temp_feature import build_feature_matrix

    BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

    print("Loading feature matrix...")

    X_scaled, session_df = build_feature_matrix(
        BASE_PATH,
        nrows=50000
    )

    print("Creating sequences...")

    window_size = 10

    X_seq, sequence_users = create_user_sequences(
        X_scaled,
        session_df,
        window_size
    )

    print("\nSequence Shape:")

    print(X_seq.shape)

    print("\nUsers Shape:")

    print(sequence_users.shape)

    print("\nSingle Sequence Shape:")

    print(X_seq[0].shape)