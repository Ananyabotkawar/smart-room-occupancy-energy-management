
"""
Shared MDP model for smart-room energy management.

Transition probabilities are estimated from observed
occupancy data. Rewards are simulated assumptions,
not measured electricity consumption.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Occupancy_Estimation.csv"

# --------------------------------------------------
# 2. States and actions
# --------------------------------------------------

STATES = {
    0: "S0 - Empty room",
    1: "S1 - One person",
    2: "S2 - Two or three people",
}

ACTIONS = {
    0: "A0 - Lights OFF, AC OFF",
    1: "A1 - Lights ON, AC LOW",
    2: "A2 - Lights ON, AC HIGH",
}

NUM_STATES = len(STATES)
NUM_ACTIONS = len(ACTIONS)
GAMMA = 0.90

# --------------------------------------------------
# 3. Estimate transition probabilities from data
# --------------------------------------------------

def load_transition_matrix():
    """Estimate P(next_state | current_state) from observations."""

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "Room_Occupancy_Count"]
    ).sort_values("timestamp").reset_index(drop=True)

    def map_occupancy(count):
        if count == 0:
            return 0
        if count == 1:
            return 1
        return 2

    df["state"] = df["Room_Occupancy_Count"].apply(map_occupancy)
    df["time_gap"] = df["timestamp"].diff().dt.total_seconds()
    df["previous_state"] = df["state"].shift(1)

    valid = (
        df["time_gap"].between(0, 45)
        & df["previous_state"].notna()
    )

    pairs = df.loc[valid, ["previous_state", "state"]]

    counts = pd.crosstab(
        pairs["previous_state"],
        pairs["state"],
    ).reindex(
        index=range(NUM_STATES),
        columns=range(NUM_STATES),
        fill_value=0,
    )

    row_totals = counts.sum(axis=1)

    if (row_totals == 0).any():
        missing_states = row_totals[row_totals == 0].index.tolist()
        raise ValueError(
            f"No valid observed transitions for states: {missing_states}"
        )

    transition_matrix = counts.div(row_totals, axis=0).to_numpy()

    return transition_matrix


TRANSITIONS = load_transition_matrix()


# Reward model: relative comfort and energy scores
# Higher reward is better.
# These are simulated assumptions, not measured energy data.
#
# Rows: S0, S1, S2
# Columns: A0, A1, A2


REWARDS = np.array([
    [  0, -2, -5],   # S0: Empty room
    [ -8, -1, -3],   # S1: One person
    [-15, -5, -2],   # S2: Two or three people
], dtype=float)


# --------------------------------------------------
# 5. Basic validation
# --------------------------------------------------

assert TRANSITIONS.shape == (NUM_STATES, NUM_STATES)
assert REWARDS.shape == (NUM_STATES, NUM_ACTIONS)
assert np.allclose(TRANSITIONS.sum(axis=1), 1.0)
assert np.all(TRANSITIONS >= 0)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("SMART ROOM MDP MODEL")
    print("=" * 40)

    print("\nStates:")
    for key, value in STATES.items():
        print(f"  {key}: {value}")

    print("\nActions:")
    for key, value in ACTIONS.items():
        print(f"  {key}: {value}")

    print("\nTransition matrix:")
    print(TRANSITIONS)

    print("\nReward matrix (illustrative assumptions):")
    print(REWARDS)

    print(f"\nDiscount factor: {GAMMA}")
    print("\nAll model checks passed.")
