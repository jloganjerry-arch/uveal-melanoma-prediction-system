import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np

def generate_km_plot(times, events, risk_scores, output_path):

    median_risk = np.median(risk_scores)

    high_risk = risk_scores >= median_risk
    low_risk = risk_scores < median_risk

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8,6))

    kmf.fit(times[low_risk], events[low_risk], label="Low Risk")
    kmf.plot_survival_function()

    kmf.fit(times[high_risk], events[high_risk], label="High Risk")
    kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()