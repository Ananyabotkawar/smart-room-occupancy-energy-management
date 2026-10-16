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


def policy_evaluation(policy, tolerance=1e-8, max_iterations=10000):
    """Evaluate a fixed policy until state values converge."""

    values = np.zeros(NUM_STATES)
    history = [values.copy()]

    for iteration in range(1, max_iterations + 1):
        new_values = np.zeros(NUM_STATES)

        for state in range(NUM_STATES):
            action = policy[state]

            new_values[state] = (
                REWARDS[state, action]
                + GAMMA * np.dot(TRANSITIONS[state], values)
            )

        difference = np.max(np.abs(new_values - values))
        values = new_values
        history.append(values.copy())

        if difference < tolerance:
            break

    else:
        raise RuntimeError("Policy evaluation did not converge.")

    return values, iteration, np.array(history)


def policy_improvement(values, old_policy):
    """Choose the best action for every state."""

    new_policy = np.zeros(NUM_STATES, dtype=int)

    for state in range(NUM_STATES):
        action_values = np.zeros(NUM_ACTIONS)

        for action in range(NUM_ACTIONS):
            action_values[action] = (
                REWARDS[state, action]
                + GAMMA * np.dot(TRANSITIONS[state], values)
            )

        new_policy[state] = int(np.argmax(action_values))

    policy_stable = np.array_equal(old_policy, new_policy)

    return new_policy, policy_stable


def policy_iteration(max_policy_iterations=1000, verbose=True):
    """Alternate policy evaluation and policy improvement."""

    # Start with A0 in every state
    policy = np.zeros(NUM_STATES, dtype=int)

    evaluation_counts = []
    evaluation_history = []
    policy_history = [policy.copy()]

    for policy_iteration_number in range(
        1, max_policy_iterations + 1
    ):
        # Phase 1: evaluate the current policy
        values, eval_iterations, history = policy_evaluation(policy)

        evaluation_counts.append(eval_iterations)
        evaluation_history.append(history)

        # Phase 2: improve the policy
        improved_policy, stable = policy_improvement(
            values, policy
        )

        if verbose:
            print(
                f"Policy iteration {policy_iteration_number}: "
                f"evaluation sweeps={eval_iterations}, "
                f"policy stable={stable}"
            )

        policy = improved_policy
        policy_history.append(policy.copy())

        # Stop only when the policy no longer changes
        if stable:
            return (
                values,
                policy,
                policy_iteration_number,
                evaluation_counts,
                evaluation_history,
                policy_history,
            )

    raise RuntimeError("Policy Iteration did not stabilize.")


def plot_policy_evaluation(history, policy_number):
    """Plot state values during one policy evaluation."""
    import matplotlib.pyplot as plt  # imported here so the app doesn't need it

    fig, ax = plt.subplots(figsize=(9, 5))

    for state in range(NUM_STATES):
        ax.plot(
            range(len(history)),
            history[:, state],
            marker="o",
            markersize=3,
            label=f"S{state}",
        )

    ax.set_title(
        f"Policy Evaluation — Policy Iteration {policy_number}"
    )
    ax.set_xlabel("Evaluation sweep")
    ax.set_ylabel("State value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / (
        f"policy_evaluation_{policy_number}.png"
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    (
        values,
        policy,
        policy_iterations,
        evaluation_counts,
        evaluation_histories,
        policy_histories,
    ) = policy_iteration()

    print("\nSMART ROOM — POLICY ITERATION")
    print("=" * 45)

    print(f"Discount factor: {GAMMA}")
    print(f"Policy improvement rounds: {policy_iterations}")
    print(f"Evaluation sweeps: {sum(evaluation_counts)}")

    print("\nFinal state values:")
    for state in range(NUM_STATES):
        print(f"  S{state}: {values[state]:.6f}")

    print("\nFinal policy:")
    for state in range(NUM_STATES):
        action = policy[state]
        print(f"  S{state} -> A{action}: {ACTIONS[action]}")

    for index, history in enumerate(evaluation_histories, start=1):
        plot_policy_evaluation(history, index)

    np.savez(
        OUTPUT_DIR / "policy_iteration_results.npz",
        values=values,
        policy=policy,
        policy_iterations=policy_iterations,
        evaluation_counts=np.array(evaluation_counts),
        policy_history=np.array(policy_histories),
    )

    print("\nResults saved to output/policy_iteration_results.npz")