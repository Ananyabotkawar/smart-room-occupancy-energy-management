
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Find the project root (smart_room_mdp/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define input and output folders
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Load original dataset from data/
DATA_PATH = DATA_DIR / "Occupancy_Estimation.csv"

# Create output folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

# Combine date and time
df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce",
)

df = df.dropna(subset=["timestamp", "Room_Occupancy_Count"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Map original occupancy counts to the three HMM states
def map_occupancy(count):
    if count == 0:
        return 0
    elif count == 1:
        return 1
    else:
        return 2

df["state"] = df["Room_Occupancy_Count"].apply(map_occupancy)
# Compare each observation with the previous one
df["time_gap"] = df["timestamp"].diff().dt.total_seconds()
df["previous_state"] = df["state"].shift(1)

# Keep only valid consecutive observations
valid = (
    df["time_gap"].between(0, 45)
    & df["previous_state"].notna()
)

pairs = df.loc[valid, ["previous_state", "state"]].copy()

# Count observed transitions
counts = pd.crosstab(
    pairs["previous_state"],
    pairs["state"],
).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)

# Normalize each row to obtain probabilities
row_totals = counts.sum(axis=1).replace(0, np.nan)
transition_matrix = counts.div(row_totals, axis=0).fillna(0)

state_names = ["S0: Empty", "S1: One person", "S2: 2–3 people"]
transition_matrix.index = state_names
transition_matrix.columns = state_names

print("\nTransition counts:")
print(counts)

print("\nTransition probability matrix:")
print(transition_matrix.round(3))

print(f"\nValid transitions used: {len(pairs)}")
print(f"Excluded time gaps: {(~valid).sum()}")

# Visualize the transition matrix
fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    transition_matrix.values,
    cmap="Blues",
    vmin=0,
    vmax=1,
)

ax.set_xticks(range(3), state_names, rotation=20, ha="right")
ax.set_yticks(range(3), state_names)

ax.set_xlabel("Next state")
ax.set_ylabel("Current state")
ax.set_title("Room Occupancy Transition Probabilities")

for i in range(3):
    for j in range(3):
        value = transition_matrix.iloc[i, j]
        text_color = "white" if value > 0.55 else "black"
        ax.text(
            j, i, f"{value:.1%}",
            ha="center", va="center",
            color=text_color, fontsize=12,
        )

fig.colorbar(image, ax=ax, label="Transition probability")
plt.tight_layout()

output_path = OUTPUT_DIR / "mdp_transition_heatmap.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nHeatmap saved to: {output_path}")
