import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os

# Set aesthetic parameters to match the UI's Royal/Academic vibe
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#c5a059', '#1e293b', '#64748b']

data_path = 'data/seer_melanoma.csv'
out_path = 'static/outputs/km_plot.png'

print(f"Reading dataset from {data_path}...")
df = pd.read_csv(data_path)

# Ensure the survival months are numeric (some might be strings like 'Unknown')
# We need to coerce errors and drop those rows
durations = pd.to_numeric(df['Survival months'], errors='coerce')

# Convert vital status to a boolean 'event observed' 
# Assuming 'Alive' means event (death) NOT observed, 'Dead' means event observed
status_col = 'Vital status recode (study cutoff used)'
events = ~df[status_col].astype(str).str.contains('Alive', case=False, na=False)

# Clean out NaNs for the fitter
valid_idx = durations.notna() & events.notna() & (durations >= 0)
T = durations[valid_idx]
E = events[valid_idx]

print(f"Fitting KM model on {len(T)} valid records...")
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
kmf.fit(T, event_observed=E, label='Overall Melanoma Population')

kmf.plot_survival_function(
    ax=ax, 
    color=colors[0], 
    linewidth=2.5,
    ci_show=True,
    ci_alpha=0.1
)

# Styling
ax.set_title('Kaplan-Meier Survival Estimate', fontname='Playfair Display', fontsize=18, fontweight='bold', color=colors[1], pad=15)
ax.set_xlabel('Timeline (Months)', fontname='Lato', fontsize=12, fontweight='bold', color=colors[2])
ax.set_ylabel('Survival Probability', fontname='Lato', fontsize=12, fontweight='bold', color=colors[2])
ax.tick_params(colors=colors[2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#e2e8f0')
ax.spines['bottom'].set_color('#e2e8f0')
ax.grid(color='#f1f5f9', linestyle='--', linewidth=1)

# Ensure outputs directory exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"Saving plot to {out_path}...")
plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', transparent=True)
print("Done.")
