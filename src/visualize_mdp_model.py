
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from mdp_model import (
    STATES,
    ACTIONS,
    TRANSITIONS,
    REWARDS,
    NUM_STATES,
    NUM_ACTIONS,
)

# --------------------------------------------------
# Paths and visual style
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "figure.facecolor": "#F5F7FB",
})

STATE_COLORS = ["#DCE8FF", "#DDF5EC", "#FFE8CC"]
EDGE_COLORS = ["#4169A1", "#208567", "#C07824"]


# --------------------------------------------------
# 1. State transition diagram
# --------------------------------------------------

def plot_transition_diagram():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#F5F7FB")
    ax.set_xlim(-1.5, 7.5)
    ax.set_ylim(-2.0, 3.0)
    ax.axis("off")

    positions = {
        0: (0.0, 0.0),
        1: (3.0, 0.0),
        2: (6.0, 0.0),
    }

    # Draw state nodes
    for state, (x, y) in positions.items():
        circle = Circle(
            (x, y),
            radius=0.72,
            facecolor=STATE_COLORS[state],
            edgecolor=EDGE_COLORS[state],
            linewidth=2.2,
            zorder=3,
        )
        ax.add_patch(circle)

        ax.text(
            x, y + 0.15, f"S{state}",
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="#172033",
            zorder=4,
        )
        ax.text(
            x, y - 0.20,
            ["Empty", "One person", "2–3 people"][state],
            ha="center", va="center",
            fontsize=9, color="#364152",
            zorder=4,
        )

    # Draw self-loops and transitions between states
    for i in range(NUM_STATES):
        x1, y1 = positions[i]

        # Self-loop above the node
        loop = FancyArrowPatch(
            (x1 - 0.35, y1 + 0.60),
            (x1 + 0.35, y1 + 0.60),
            connectionstyle="arc3,rad=-1.8",
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=EDGE_COLORS[i],
            zorder=2,
        )
        ax.add_patch(loop)

        ax.text(
            x1, 1.35,
            f"{TRANSITIONS[i, i]:.2%}",
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color=EDGE_COLORS[i],
        )

    # Draw transitions between different states.
    # Very small probabilities are omitted visually to reduce clutter.
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            if i == j or TRANSITIONS[i, j] < 0.005:
                continue

            x1, y1 = positions[i]
            x2, y2 = positions[j]
            direction = 1 if j > i else -1

            arrow = FancyArrowPatch(
                (x1 + direction * 0.65, y1 + 0.25),
                (x2 - direction * 0.65, y2 + 0.25),
                connectionstyle=f"arc3,rad={0.18 * direction}",
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8,
                color="#64748B",
                zorder=1,
            )
            ax.add_patch(arrow)

            ax.text(
                (x1 + x2) / 2,
                0.55 + 0.22 * direction,
                f"{TRANSITIONS[i, j]:.2%}",
                ha="center", va="center",
                fontsize=9, color="#172033",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#FFFFFF",
                    edgecolor="none",
                    alpha=0.95,
                ),
            )

    ax.set_title(
        "Smart Room: Occupancy State Transitions",
        fontweight="bold",
        pad=18,
        color="#172033",
    )

    ax.text(
        3, -1.35,
        "Probabilities are estimated from observed occupancy data.\n"
        "Transitions below 0.5% are hidden to keep the diagram readable.",
        ha="center", va="center",
        fontsize=10, color="#475569",
    )

    plt.tight_layout()
    path = OUTPUT_DIR / "mdp_state_transition_diagram.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------
# 2. Reward matrix heatmap
# --------------------------------------------------

def plot_reward_heatmap():
    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(REWARDS, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(NUM_ACTIONS))
    ax.set_xticklabels(
        [f"A{i}\n{ACTIONS[i].split(' - ')[1]}" for i in ACTIONS]
    )
    ax.set_yticks(range(NUM_STATES))
    ax.set_yticklabels(
        [STATES[i] for i in STATES]
    )

    ax.set_xlabel("Control action")
    ax.set_ylabel("Occupancy state")
    ax.set_title(
        "MDP Reward Matrix",
        fontweight="bold",
        pad=15,
    )

    for i in range(NUM_STATES):
        for j in range(NUM_ACTIONS):
            value = REWARDS[i, j]
            ax.text(
                j, i, f"{value:.1f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="#172033",
            )

    fig.colorbar(image, ax=ax, label="Reward (higher is better)")
    ax.text(
        0.5, -0.22,
        "Illustrative design assumptions, not measured energy savings.",
        transform=ax.transAxes,
        ha="center", fontsize=9, color="#64748B",
    )

    plt.tight_layout()
    path = OUTPUT_DIR / "mdp_reward_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------
# 3. Transition probability heatmap
# --------------------------------------------------

def plot_transition_heatmap():
    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        TRANSITIONS,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    labels = ["S0: Empty", "S1: One person", "S2: 2–3 people"]
    ax.set_xticks(range(NUM_STATES), labels, rotation=20, ha="right")
    ax.set_yticks(range(NUM_STATES), labels)

    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title(
        "MDP Transition Probability Matrix",
        fontweight="bold",
        pad=15,
    )

    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            value = TRANSITIONS[i, j]
            ax.text(
                j, i, f"{value:.2%}",
                ha="center", va="center",
                color="white" if value > 0.55 else "#172033",
                fontsize=11,
            )

    fig.colorbar(image, ax=ax, label="Transition probability")
    plt.tight_layout()

    path = OUTPUT_DIR / "mdp_transition_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------
# Run all visualizations
# --------------------------------------------------

if __name__ == "__main__":
    plot_transition_diagram()
    plot_reward_heatmap()
    plot_transition_heatmap()

    print("\nAll MDP visualizations generated successfully.")
