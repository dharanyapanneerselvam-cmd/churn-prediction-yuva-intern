"""
Final Evaluation & Presentation Summary
Week 5 Task - Yuva Intern (Data Science Project Coordinator)
Author: Dharanya

Consolidates outcomes from all project phases (Weeks 1-4) into a
single final evaluation summary for stakeholder presentation.
"""

from datetime import datetime


PROJECT_PHASES = {
    "Week 1: Planning & Strategy": {
        "outcome": "Defined project scope, 5-phase plan, risk register, and Gantt timeline",
        "challenge": "Balancing realistic timelines with limited resource assumptions",
    },
    "Week 2: Data Acquisition & Management": {
        "outcome": "Identified 4 data sources; built a 6-step Python cleaning pipeline",
        "challenge": "Ensuring ethical handling of customer-identifiable data",
    },
    "Week 3: Execution & Implementation": {
        "outcome": "Trained and validated Logistic Regression & Random Forest models via Agile sprints",
        "challenge": "Coordinating model iteration within tight sprint cycles",
    },
    "Week 4: Monitoring & Reporting": {
        "outcome": "Defined 5 KPIs, built a Python-based dashboard and corrective-action cycle",
        "challenge": "Setting realistic, evidence-based KPI targets",
    },
}

FINAL_METRICS = {
    "accuracy": 0.83,
    "precision": 0.77,
    "recall": 0.68,
    "f1_score": 0.72,
}


def review_all_phases() -> dict:
    """Print and return a structured summary of every project phase."""
    for phase, details in PROJECT_PHASES.items():
        print(f"\n{phase}")
        print(f"  Outcome:   {details['outcome']}")
        print(f"  Challenge: {details['challenge']}")
    return PROJECT_PHASES


def summarize_success_metrics(metrics: dict = FINAL_METRICS) -> str:
    """Summarize final model performance against project KPIs."""
    lines = [f"{k}: {v:.2f}" for k, v in metrics.items()]
    summary = "Final Model Performance -> " + ", ".join(lines)
    print(summary)
    return summary


def lessons_learned() -> list:
    """Return the consolidated lessons-learned list for the final report."""
    lessons = [
        "Early stakeholder alignment on KPI targets avoided rework in later sprints.",
        "Public datasets accelerated prototyping while internal-data approval was pending.",
        "Automated KPI dashboards made deviation detection faster than manual review.",
        "Retaining an interpretable baseline model (Logistic Regression) helped stakeholder trust.",
    ]
    for l in lessons:
        print("-", l)
    return lessons


def generate_final_report():
    """Combine all sections into one final evaluation report (for the DOC)."""
    print(f"FINAL PROJECT EVALUATION REPORT - {datetime.now().date()}")
    review_all_phases()
    summarize_success_metrics()
    lessons_learned()


if __name__ == "__main__":
    generate_final_report()
