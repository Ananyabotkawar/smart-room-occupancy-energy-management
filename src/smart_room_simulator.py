
from pathlib import Path
import pandas as pd

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the HMM prediction dataset
DATA_FILE = PROJECT_DIR / "data" / "hmm_predictions.csv"

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Columns required by the simulator
required_columns = [
    "timestamp",
    "P(S0)",
    "P(S1)",
    "P(S2)"
]

# Check that all required columns exist
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Check for missing belief probabilities
if df[["P(S0)", "P(S1)", "P(S2)"]].isnull().any().any():
    raise ValueError("Missing values found in belief probabilities.")

# Verify that probabilities sum to approximately 1
probability_sums = df[["P(S0)", "P(S1)", "P(S2)"]].sum(axis=1)

if not probability_sums.between(0.99, 1.01).all():
    raise ValueError("Some belief probabilities do not sum to 1.")

print("HMM dataset loaded successfully!")
print(f"Total records: {len(df)}")
print("\nFirst five records:")
print(df[required_columns].head())

print("\nProbability validation:")
print(f"Minimum probability sum: {probability_sums.min():.6f}")
print(f"Maximum probability sum: {probability_sums.max():.6f}")


import numpy as np

# Occupancy states and control actions
STATES = ["S0", "S1", "S2"]

ACTIONS = {
    0: "A0 - Lights OFF, AC OFF",
    1: "A1 - Lights ON, AC LOW",
    2: "A2 - Lights ON, AC HIGH"
}

# Must match REWARDS in src/mdp_model.py
REWARDS = np.array([
    [  0, -2, -5],   # S0: Empty room
    [ -8, -1, -3],   # S1: One person
    [-15, -5, -2],   # S2: Two or three people
], dtype=float)

# Extract HMM belief probabilities
beliefs = df[["P(S0)", "P(S1)", "P(S2)"]].to_numpy()

# Expected immediate reward for every action
expected_rewards = beliefs @ REWARDS

# Choose the action with the highest expected reward
selected_actions = np.argmax(expected_rewards, axis=1)

# Store simulation results
df["expected_reward_A0"] = expected_rewards[:, 0]
df["expected_reward_A1"] = expected_rewards[:, 1]
df["expected_reward_A2"] = expected_rewards[:, 2]

df["selected_action"] = selected_actions
df["action_name"] = df["selected_action"].map(ACTIONS)

# Display the first five decisions
print("\nFirst five simulated decisions:")
print(
    df[[
        "timestamp",
        "P(S0)", "P(S1)", "P(S2)",
        "action_name"
    ]].head().to_string(index=False)
)

print("\nAction selection counts:")
print(df["action_name"].value_counts())
