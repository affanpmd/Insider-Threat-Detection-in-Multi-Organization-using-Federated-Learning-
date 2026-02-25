import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

BASE_PATH = r"C:\Users\affan\Downloads\r5.2\r5.2"

logon = pd.read_csv(f"{BASE_PATH}\\logon.csv", nrows=100000)
device = pd.read_csv(f"{BASE_PATH}\\device.csv", nrows=100000)
http = pd.read_csv(f"{BASE_PATH}\\http.csv", nrows=100000)
email = pd.read_csv(f"{BASE_PATH}\\email.csv", nrows=100000)
file_df = pd.read_csv(f"{BASE_PATH}\\file.csv", nrows=100000)
decoy = pd.read_csv(f"{BASE_PATH}\\decoy_file.csv", nrows=100000)
psychometric = pd.read_csv(f"{BASE_PATH}\\psychometric.csv")

# Load LDAP
ldap_files = glob.glob(f"{BASE_PATH}\\LDAP\\*.csv")
ldap = pd.concat([pd.read_csv(f) for f in ldap_files], ignore_index=True)

# =====================================================
# 2️⃣ DATETIME PROCESSING
# =====================================================

for df in [logon, device, http, email, file_df]:
    df["date"] = pd.to_datetime(df["date"])

logon["hour"] = logon["date"].dt.hour
device["hour"] = device["date"].dt.hour
http["hour"] = http["date"].dt.hour
email["hour"] = email["date"].dt.hour

# =====================================================
# 3️⃣ PRE-GROUP BY USER  (CRITICAL OPTIMIZATION)
# =====================================================

device_group = dict(tuple(device.groupby("user")))
http_group = dict(tuple(http.groupby("user")))
email_group = dict(tuple(email.groupby("user")))
file_group = dict(tuple(file_df.groupby("user")))
psych_group = dict(tuple(psychometric.groupby("user_id")))

# LDAP
ldap["admin_privilege"] = (ldap["role"] == "ITAdmin").astype(int)
org_features = ldap[["user_id", "admin_privilege"]].drop_duplicates()
org_features.rename(columns={"user_id": "user"}, inplace=True)

# Decoy filename set (fast lookup)
decoy_col = decoy.columns[0]
decoy_set = set(decoy[decoy_col])

# Get correct filename column from file.csv
file_col = file_df.columns[file_df.columns.str.contains("file", case=False)][0]

# =====================================================
# 4️⃣ SESSION CONSTRUCTION
# =====================================================

users = logon["user"].unique()

sessions = []

for user in users:

    user_logs = logon[logon["user"] == user].sort_values("date").reset_index(drop=True)

    login_hours = user_logs[user_logs["activity"] == "Logon"]["hour"]
    avg_login = login_hours.mean() if not login_hours.empty else 0

    # Filter ONCE per user (not globally grouped)
    user_device = device[device["user"] == user]
    user_http = http[http["user"] == user]
    user_email = email[email["user"] == user]
    user_file = file_df[file_df["user"] == user]
    user_psych = psychometric[psychometric["user_id"] == user]

    for i in range(len(user_logs) - 1):

        if user_logs.loc[i, "activity"] == "Logon" and \
           user_logs.loc[i+1, "activity"] == "Logoff":

            start = user_logs.loc[i, "date"]
            end = user_logs.loc[i+1, "date"]

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

            # FILE
            file_events = user_file[
                (user_file["date"] >= start) &
                (user_file["date"] <= end)
            ]
            file_copy_count = len(file_events)

            # Skip decoy temporarily (to reduce memory)
            decoy_access = 0

            # WEB
            web_events = user_http[
                (user_http["date"] >= start) &
                (user_http["date"] <= end)
            ]
            web_count = len(web_events)
            after_hours_web = int(
                any((web_events["hour"] < 8) | (web_events["hour"] > 18))
            )

            # EMAIL
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
                decoy_access,
                web_count,
                after_hours_web,
                emails_sent,
                attachment_count,
                external_ratio,
                O, C, E, A, N
            ])


# =====================================================
# 5️⃣ BUILD Xt
# =====================================================

columns = [
    "user","login_hour","after_hours_login","weekend_flag",
    "session_duration","deviation_login",
    "usb_used","file_copy_count","decoy_access",
    "web_count","after_hours_web",
    "emails_sent","attachment_count","external_ratio",
    "O","C","E","A","N"
]

session_df = pd.DataFrame(sessions, columns=columns)
session_df = session_df.merge(org_features, how="left", on="user")
session_df = session_df.fillna(0)

feature_cols = session_df.columns.drop("user")

X = session_df[feature_cols].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Xt Shape:", X_scaled.shape)
print("First 5 Xt rows:\n", X_scaled[:5])

