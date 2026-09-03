"""
Monitoring, Reporting & Process Improvement Module
Week 4 Task - Yuva Intern (Data Science Project Coordinator)
Author: Dharanya
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


KPI_TARGETS = {
    "accuracy": 0.80,
    "precision": 0.75,
    "recall": 0.70,
    "f1": 0.72,
    "data_freshness_days": 7,
}


def log_snapshot(metrics: dict, log_path: str = "monitoring_log.csv"):
    row = {"timestamp": datetime.now().isoformat(), **metrics}
    df_row = pd.DataFrame([row])
    try:
        existing = pd.read_csv(log_path)
        df_row = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        pass
    df_row.to_csv(log_path, index=False)
    print(f"Snapshot logged to {log_path}")
    return df_row


def check_kpi_deviation(metrics: dict) -> dict:
    deviations = {}
    for kpi, target in KPI_TARGETS.items():
        if kpi in metrics:
            if kpi == "data_freshness_days":
                deviations[kpi] = metrics[kpi] > target
            else:
                deviations[kpi] = metrics[kpi] < target
    flagged = {k: v for k, v in deviations.items() if v}
    if flagged:
        print("KPIs needing corrective action:", list(flagged.keys()))
    else:
        print("All KPIs within target range.")
    return flagged


def generate_dashboard(log_path: str = "monitoring_log.csv", out_path: str = "kpi_dashboard.png"):
    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    metrics_to_plot = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df.columns]
    plt.figure(figsize=(9, 5))
    for m in metrics_to_plot:
        plt.plot(df["timestamp"], df[m], marker="o", label=m)
        plt.axhline(y=KPI_TARGETS.get(m, 0), linestyle="--", alpha=0.4)
    plt.title("Churn Model KPI Trend Dashboard")
    plt.xlabel("Report Date")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Dashboard saved to {out_path}")


def generate_weekly_report(metrics: dict, stakeholders=None) -> str:
    stakeholders = stakeholders or ["Project Coordinator", "Data Scientist", "Sponsor"]
    flagged = check_kpi_deviation(metrics)
    status = "ON TRACK" if not flagged else "NEEDS CORRECTIVE ACTION"
    report = (
        f"Weekly Churn Model Performance Report ({datetime.now().date()})\n"
        f"Status: {status}\n"
        f"Metrics: {metrics}\n"
        f"Flagged KPIs: {list(flagged.keys()) if flagged else 'None'}\n"
        f"Distributed to: {', '.join(stakeholders)}"
    )
    print(report)
    return report


if __name__ == "__main__":
    latest_metrics = {"accuracy": 0.83, "precision": 0.77, "recall": 0.66, "f1": 0.71, "data_freshness_days": 3}
    log_snapshot(latest_metrics)
    generate_weekly_report(latest_metrics)
