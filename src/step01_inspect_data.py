

from pathlib import Path
import pandas as pd

# Find the project root (smart_room_mdp/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the data folder
DATA_DIR = PROJECT_ROOT / "data"

# Load the HMM predictions
HMM_PATH = DATA_DIR / "hmm_predictions.csv"
df = pd.read_csv(HMM_PATH)


# Display the basic structure
print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

# Check the probability columns
probability_columns = ["P(S0)", "P(S1)", "P(S2)"]

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Probability sums (should be approximately 1) ---")
print(df[probability_columns].sum(axis=1).describe())

print("\n--- Predicted state counts ---")
print(df["predicted_state"].value_counts())

print("\n--- Actual state counts ---")
print(df["true_state"].value_counts())



# Inspect the time intervals between observations
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")
time_gaps = df["timestamp"].diff().dt.total_seconds().dropna()

print("\n--- Time gaps between observations (seconds) ---")
print(time_gaps.describe())

print("\n--- Most common time gaps ---")
print(time_gaps.value_counts().head(10))

print("\n--- First and last timestamps ---")
print(df["timestamp"].min())
print(df["timestamp"].max())
