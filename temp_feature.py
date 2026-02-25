# feature_engineering.py

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler


def build_feature_matrix(base_path, nrows=100000):
    """
    Builds scaled session feature matrix.

    Returns:
        X_scaled : numpy array (num_sessions, num_features)
        session_df : dataframe including 'user'
    """

    # =====================================================
    # 1️⃣ LOAD DATA
    # =====================================================

    logon = pd.read_csv(f"{base_path}\\logon.csv", nrows=nrows)
    device = pd.read_csv(f"{base_path}\\device.csv", nrows=nrows)
    http = pd.read_csv(f"{base_path}\\http.csv", nrows=nrows)
    email = pd.read_csv(f"{base_path}\\email.csv", nrows=nrows)
    file_df = pd.read_csv(f"{base_path}\\file.csv", nrows=nrows)
    psychometric = pd.read_csv(f"{base_path}\\psychometric.csv")

    # Load LDAP
    ldap_files = glob.glob(f"{base_path}\\LDAP\\*.csv")
    ldap = pd.concat([pd.read_csv(f) for f in ldap_files], ignore_index=True)

    # =====================================================
    # 2️⃣ DATETIME PROCESSING
    # =====================================================

    for df in [logon, device, http, email, file_df]:
        df["date"] = pd.to_datetime(df["date"])

    logon["hour"] = logon["date"].dt.hour
    http["hour"] = http["date"].dt.hour

    # =====================================================
    # 3️⃣ ORG FEATURES
    # =====================================================

    ldap["admin_privilege"] = (ldap["role"] == "ITAdmin").astype(int)
    org_features = ldap[["user_id", "admin_privilege"]].drop_duplicates()
    org_features.rename(columns={"user_id": "user"}, inplace=True)

    # =====================================================
    # 4️⃣ SESSION CONSTRUCTION
    # =====================================================

    sessions = []

    for user in logon["user"].unique():

        user_logs = logon[logon["user"] == user].sort_values("date").reset_index(drop=True)

        login_hours = user_logs[user_logs["activity"] == "Logon"]["hour"]
        avg_login = login_hours.mean() if not login_hours.empty else 0

        user_device = device[device["user"] == user]
        user_http = http[http["user"] == user]
        user_email = email[email["user"] == user]
        user_file = file_df[file_df["user"] == user]
        user_psych = psychometric[psychometric["user_id"] == user]

        for i in range(len(user_logs) - 1):

            if user_logs.loc[i, "activity"] == "Logon" and \
               user_logs.loc[i + 1, "activity"] == "Logoff":

                start = user_logs.loc[i, "date"]
                end = user_logs.loc[i + 1, "date"]

                login_hour = start.hour
                after_hours_login = int(login_hour < 8 or login_hour > 18)
                weekend_flag = int(start.weekday() >= 5)
                session_duration = (end - start).total_seconds() / 3600
                deviation_login = abs(login_hour - avg_login)

                # USB
                usb_events = user_device[
                    (user_device["date"] >= start) &
                    (user_device["date"] <= end) &
                    (user_device["activity"] == "connect")
                ]
                usb_used = int(len(usb_events) > 0)

                # File activity
                file_events = user_file[
                    (user_file["date"] >= start) &
                    (user_file["date"] <= end)
                ]
                file_copy_count = len(file_events)

                # Web activity
                web_events = user_http[
                    (user_http["date"] >= start) &
                    (user_http["date"] <= end)
                ]
                web_count = len(web_events)
                after_hours_web = int(
                    any((web_events["hour"] < 8) | (web_events["hour"] > 18))
                )

                # Email activity
                email_events = user_email[
                    (user_email["date"] >= start) &
                    (user_email["date"] <= end)
                ]
                emails_sent = len(email_events[email_events["activity"] == "send"])

                attachment_count = email_events["attachments"].fillna("").apply(
                    lambda x: 0 if x == "" else len(str(x).split(";"))
                ).sum()

                external_ratio = 0
                if len(email_events) > 0:
                    external = email_events["to"].fillna("").str.contains("@").sum()
                    external_ratio = external / len(email_events)

                # Psychometric
                if not user_psych.empty:
                    row = user_psych.iloc[0]
                    O, C, E, A, N = row["O"], row["C"], row["E"], row["A"], row["N"]
                else:
                    O = C = E = A = N = 0

                sessions.append([
                    user,
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

    columns = [
        "user", "login_hour", "after_hours_login", "weekend_flag",
        "session_duration", "deviation_login",
        "usb_used", "file_copy_count",
        "web_count", "after_hours_web",
        "emails_sent", "attachment_count", "external_ratio",
        "O", "C", "E", "A", "N"
    ]

    session_df = pd.DataFrame(sessions, columns=columns)
    session_df = session_df.merge(org_features, how="left", on="user")
    session_df = session_df.fillna(0)

    feature_cols = session_df.columns.drop("user")
    X = session_df[feature_cols].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, session_df


if __name__ == "__main__":

    BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

    X_scaled, session_df = build_feature_matrix(BASE_PATH)

    print("Xt Shape:", X_scaled.shape)
    print("First 5 rows:\n", X_scaled[:5])