import numpy as np


print("Loading model predictions...")

lstm_pred = np.load("lstm_predictions.npy")
fedavg_pred = np.load("fedavg_predictions.npy")
weighted_pred = np.load("weighted_fedavg_predictions.npy")


def evaluate_model(name, preds):

    total_sequences = len(preds)
    anomalies = np.sum(preds)

    anomaly_rate = anomalies / total_sequences

    print("\n==============================")
    print(name)
    print("==============================")

    print("Total Sequences :", total_sequences)
    print("Detected Anomalies :", anomalies)
    print("Anomaly Rate :", round(anomaly_rate, 5))


evaluate_model("Centralized LSTM", lstm_pred)
evaluate_model("FedAvg", fedavg_pred)
evaluate_model("Weighted FedAvg", weighted_pred)