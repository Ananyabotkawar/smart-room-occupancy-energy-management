from pathlib import Path

import numpy as np

from mdp_model import (
    STATES,
    ACTIONS,
    TRANSITIONS,
    REWARDS,
    NUM_STATES,
    NUM_ACTIONS,
    GAMMA,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def value_iteration(tolerance=1e-8, max_iterations=10000, verbose=True):
    values = np.zeros(NUM_STATES)
    if verbose:
        print("Initial state values:", values)
    history = [values.copy()]

    for iteration in range(1, max_iterations + 1):
        new_values = np.zeros(NUM_STATES)

        for state in range(NUM_STATES):
            action_values = []

            for action in range(NUM_ACTIONS):
                # Expected next-state value
                future_value = np.dot(TRANSITIONS[state], values)

                # Bellman optimality update
                q_value = REWARDS[state, action] + GAMMA * future_value
                action_values.append(q_value)

            new_values[state] = max(action_values)

        difference = np.max(np.abs(new_values - values))
        values = new_values
        if verbose:
            print(f"Iteration {iteration}: {values}")
        history.append(values.copy())

        if difference < tolerance:
            break

    # Extract the greedy policy from converged state values
    policy = np.zeros(NUM_STATES, dtype=int)

    for state in range(NUM_STATES):
        q_values = [
            REWARDS[state, action]
            + GAMMA * np.dot(TRANSITIONS[state], values)
            for action in range(NUM_ACTIONS)
        ]
        policy[state] = int(np.argmax(q_values))

    if iteration == max_iterations and difference >= tolerance:
        raise RuntimeError("Value Iteration did not converge.")

    return values, policy, iteration, np.array(history)


def plot_convergence(history):
    import matplotlib.pyplot as plt  # imported here so the app doesn't need it

    fig, ax = plt.subplots(figsize=(9, 5))

    for state in range(NUM_STATES):
        ax.plot(
            range(len(history)),
            history[:, state],
            marker="o",
            markersize=3,
            label=STATES[state],
        )

    ax.set_title("Value Iteration: State Value Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated state value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "value_iteration_convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence graph saved: {path}")


if __name__ == "__main__":
    values, policy, iterations, history = value_iteration()

    print("\nSMART ROOM — VALUE ITERATION")
    print("=" * 45)
    print(f"Discount factor: {GAMMA}")
    print(f"Iterations: {iterations}")

    print("\nOptimal state values:")
    for state in range(NUM_STATES):
        print(f"  {STATES[state]}: {values[state]:.4f}")

    print("\nRecommended control policy:")
    for state in range(NUM_STATES):
        print(f"  {STATES[state]} -> {ACTIONS[policy[state]]}")

    plot_convergence(history)

    np.savez(
        OUTPUT_DIR / "value_iteration_results.npz",
        values=values,
        policy=policy,
        iterations=iterations,
        history=history,
    )
    print("Results saved in output/value_iteration_results.npz")