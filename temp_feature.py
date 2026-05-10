# feature_engineering.py

import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import MinMaxScaler


# =====================================================
# BUILD FEATURE MATRIX
# =====================================================

def build_feature_matrix(base_path, nrows=100000):

    """
    Builds scaled session feature matrix.

    Parameters:
        base_path : CERT dataset path
        nrows     : number of rows to load dynamically

    Returns:
        X_scaled  : scaled feature matrix
        session_df: dataframe containing sessions + users
    """

    print(f"\nLoading {nrows} rows from dataset...")

    # =====================================================
    # 1️⃣ LOAD DATA
    # =====================================================

    logon = pd.read_csv(
        f"{base_path}\\logon.csv",
        nrows=nrows
    )

    device = pd.read_csv(
        f"{base_path}\\device.csv",
        nrows=nrows
    )

    http = pd.read_csv(
        f"{base_path}\\http.csv",
        nrows=nrows
    )

    email = pd.read_csv(
        f"{base_path}\\email.csv",
        nrows=nrows
    )

    file_df = pd.read_csv(
        f"{base_path}\\file.csv",
        nrows=nrows
    )

    psychometric = pd.read_csv(
        f"{base_path}\\psychometric.csv"
    )

    # =====================================================
    # 2️⃣ LOAD LDAP
    # =====================================================

    ldap_files = glob.glob(f"{base_path}\\LDAP\\*.csv")

    ldap = pd.concat(
        [pd.read_csv(f) for f in ldap_files],
        ignore_index=True
    )

    # =====================================================
    # 3️⃣ DATETIME PROCESSING
    # =====================================================

    for df in [logon, device, http, email, file_df]:

        df["date"] = pd.to_datetime(
            df["date"],
            errors='coerce'
        )

    # Drop invalid rows
    logon = logon.dropna(subset=["date"])

    # Extract hours
    logon["hour"] = logon["date"].dt.hour
    http["hour"] = http["date"].dt.hour

    # =====================================================
    # 4️⃣ ORG FEATURES
    # =====================================================

    ldap["admin_privilege"] = (
        ldap["role"] == "ITAdmin"
    ).astype(int)

    org_features = ldap[
        ["user_id", "admin_privilege"]
    ].drop_duplicates()

    org_features.rename(
        columns={"user_id": "user"},
        inplace=True
    )

    # =====================================================
    # 5️⃣ SESSION CONSTRUCTION
    # =====================================================

    sessions = []

    users = logon["user"].unique()

    print("Building user sessions...")

    for user in users:

        user_logs = logon[
            logon["user"] == user
        ].sort_values("date").reset_index(drop=True)

        if len(user_logs) < 2:
            continue

        login_hours = user_logs[
            user_logs["activity"] == "Logon"
        ]["hour"]

        avg_login = (
            login_hours.mean()
            if not login_hours.empty
            else 0
        )

        # User-specific activity
        user_device = device[device["user"] == user]
        user_http = http[http["user"] == user]
        user_email = email[email["user"] == user]
        user_file = file_df[file_df["user"] == user]
        user_psych = psychometric[
            psychometric["user_id"] == user
        ]

        # =================================================
        # SESSION LOOP
        # =================================================

        for i in range(len(user_logs) - 1):

            current_activity = user_logs.loc[i, "activity"]
            next_activity = user_logs.loc[i + 1, "activity"]

            if current_activity == "Logon" and \
               next_activity == "Logoff":

                start = user_logs.loc[i, "date"]
                end = user_logs.loc[i + 1, "date"]

                # Skip invalid sessions
                if end <= start:
                    continue

                # =========================================
                # FEATURE EXTRACTION
                # =========================================

                login_hour = start.hour

                after_hours_login = int(
                    login_hour < 8 or login_hour > 18
                )

                weekend_flag = int(
                    start.weekday() >= 5
                )

                session_duration = (
                    (end - start).total_seconds() / 3600
                )

                deviation_login = abs(
                    login_hour - avg_login
                )

                # =========================================
                # USB ACTIVITY
                # =========================================

                usb_events = user_device[
                    (user_device["date"] >= start) &
                    (user_device["date"] <= end) &
                    (user_device["activity"] == "connect")
                ]

                usb_used = int(len(usb_events) > 0)

                # =========================================
                # FILE ACTIVITY
                # =========================================

                file_events = user_file[
                    (user_file["date"] >= start) &
                    (user_file["date"] <= end)
                ]

                file_copy_count = len(file_events)

                # =========================================
                # WEB ACTIVITY
                # =========================================

                web_events = user_http[
                    (user_http["date"] >= start) &
                    (user_http["date"] <= end)
                ]

                web_count = len(web_events)

                after_hours_web = int(
                    any(
                        (web_events["hour"] < 8) |
                        (web_events["hour"] > 18)
                    )
                )

                # =========================================
                # EMAIL ACTIVITY
                # =========================================

                email_events = user_email[
                    (user_email["date"] >= start) &
                    (user_email["date"] <= end)
                ]

                emails_sent = len(
                    email_events[
                        email_events["activity"] == "send"
                    ]
                )

                attachment_count = email_events[
                    "attachments"
                ].fillna("").apply(
                    lambda x:
                    0 if x == ""
                    else len(str(x).split(";"))
                ).sum()

                external_ratio = 0

                if len(email_events) > 0:

                    external = email_events[
                        "to"
                    ].fillna("").str.contains("@").sum()

                    external_ratio = external / len(email_events)

                # =========================================
                # PSYCHOMETRIC FEATURES
                # =========================================

                if not user_psych.empty:

                    row = user_psych.iloc[0]

                    O = row["O"]
                    C = row["C"]
                    E = row["E"]
                    A = row["A"]
                    N = row["N"]

                else:
                    O = C = E = A = N = 0

                # =========================================
                # SAVE SESSION
                # =========================================

                sessions.append([

                    user,

                    # session timing
                    start,
                    end,

                    login_hour,
                    after_hours_login,
                    weekend_flag,
                    session_duration,
                    deviation_login,

                    usb_used,
                    file_copy_count,

                    web_count,
                    after_hours_web,

                    emails_sent,
                    attachment_count,
                    external_ratio,

                    O, C, E, A, N
                ])

    # =====================================================
    # 6️⃣ BUILD DATAFRAME
    # =====================================================

    columns = [

        "user",

        # session timing
        "session_start",
        "session_end",

        "login_hour",
        "after_hours_login",
        "weekend_flag",
        "session_duration",
        "deviation_login",

        "usb_used",
        "file_copy_count",

        "web_count",
        "after_hours_web",

        "emails_sent",
        "attachment_count",
        "external_ratio",

        "O", "C", "E", "A", "N"
    ]

    session_df = pd.DataFrame(
        sessions,
        columns=columns
    )

    # =====================================================
    # MERGE ORG FEATURES
    # =====================================================

    session_df = session_df.merge(
        org_features,
        how="left",
        on="user"
    )

    session_df = session_df.fillna(0)

    # =====================================================
    # 7️⃣ SCALE FEATURES
    # =====================================================

    feature_cols = session_df.columns.drop([
        "user",
        "session_start",
        "session_end"
    ])

    X = session_df[feature_cols].values

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    print("Feature Matrix Shape:", X_scaled.shape)

    return X_scaled, session_df


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

    X_scaled, session_df = build_feature_matrix(
        BASE_PATH,
        nrows=50000
    )

    print("\nFirst 5 Feature Rows:\n")

    print(session_df.head())

    print("\nScaled Matrix Shape:")

    print(X_scaled.shape)