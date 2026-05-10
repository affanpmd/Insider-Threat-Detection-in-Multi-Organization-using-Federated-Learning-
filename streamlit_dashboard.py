# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import time
import os


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Insider Threat Detection Dashboard",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title(
    "🔐 Real-Time Insider Threat Detection Dashboard"
)

st.markdown(
    """
    Monitoring insider threats using:

    - LSTM Autoencoder
    - Weighted FedAvg
    - Real-Time Behavioral Analysis
    """
)


# =====================================================
# CSV FILE
# =====================================================

csv_file = "realtime_results.csv"


# =====================================================
# REFRESH SETTINGS
# =====================================================

refresh_interval = 15

placeholder = st.empty()


# =====================================================
# LIVE DASHBOARD LOOP
# =====================================================

while True:

    with placeholder.container():

        st.subheader(
            "📡 Live Monitoring Results"
        )

        # =============================================
        # CHECK FILE
        # =============================================

        if not os.path.exists(csv_file):

            st.warning(
                "CSV file not found."
            )

            st.info(
                "Run realtime_monitor.py first."
            )

        else:

            # =========================================
            # LOAD DATA
            # =========================================

            df = pd.read_csv(csv_file)

            # latest row
            latest = df.iloc[-1]

            # =========================================
            # METRICS
            # =========================================

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Sequences",
                int(latest["Total Sequences"])
            )

            col2.metric(
                "Detected Anomalies",
                int(latest["Detected Anomalies"])
            )

            col3.metric(
                "Anomaly Rate",
                float(latest["Anomaly Rate"])
            )

            col4.metric(
                "Top Risky User",
                str(latest["Top Risky User"])
            )

            st.markdown("---")

            # =========================================
            # ALERT PANEL
            # =========================================

            st.subheader(
                "🚨 Suspicious Activity Alert"
            )

            st.error(
                f"""
                User:
                {latest['Top Risky User']}

                Activity:
                {latest['Suspicious Activity']}

                Session Time:
                {latest['Suspicious Session Time']}
                """
            )

            st.markdown("---")

            # =========================================
            # ANOMALY TREND
            # =========================================

            st.subheader(
                "📈 Detected Anomalies Trend"
            )

            chart_data = df[
                ["Detected Anomalies"]
            ]

            st.line_chart(chart_data)

            # =========================================
            # ANOMALY RATE TREND
            # =========================================

            st.subheader(
                "📊 Anomaly Rate Trend"
            )

            rate_data = df[
                ["Anomaly Rate"]
            ]

            st.area_chart(rate_data)

            st.markdown("---")

            # =========================================
            # RECENT ALERTS TABLE
            # =========================================

            st.subheader(
                "📝 Recent Monitoring Logs"
            )

            st.dataframe(
                df.tail(20),
                use_container_width=True
            )

            st.markdown("---")

            # =========================================
            # TOP INSIDER ALERT
            # =========================================

            st.subheader(
                "⚠️ Current Highest Risk User"
            )

            st.warning(
                f"""
                {latest['Top Risky User']}
                is currently showing
                anomalous behavioral patterns.
                """
            )

            # =========================================
            # LAST UPDATE
            # =========================================

            st.success(
                f"""
                Last Updated:
                {latest['Timestamp']}
                """
            )

    # =================================================
    # AUTO REFRESH
    # =================================================

    time.sleep(refresh_interval)