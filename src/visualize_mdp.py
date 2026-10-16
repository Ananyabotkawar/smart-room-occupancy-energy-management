
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Save all generated visuals inside output/
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Our three occupancy states
states = [
    ("S0", "Empty room", "0 people"),
    ("S1", "One person", "1 person"),
    ("S2", "Higher occupancy", "2–3 people"),
]

# Our three possible control actions
actions = [
    ("A0", "Lights OFF", "AC OFF"),
    ("A1", "Lights ON", "AC LOW"),
    ("A2", "Lights ON", "AC HIGH"),
]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#F5F7FB")
ax.set_facecolor("#F5F7FB")
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

def draw_box(x, y, width, height, title, subtitle,
             facecolor, edgecolor):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2, y + height * 0.65, title,
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#172033",
    )
    ax.text(
        x + width / 2, y + height * 0.30, subtitle,
        ha="center", va="center",
        fontsize=10, color="#364152",
    )

ax.text(
    6, 7.45, "SMART ROOM MDP: STATES AND ACTIONS",
    ha="center", fontsize=19, fontweight="bold",
    color="#172033",
)

ax.text(
    3, 6.65, "STATES — WHAT IS THE ROOM LIKE?",
    ha="center", fontsize=12, fontweight="bold",
    color="#2457A7",
)

ax.text(
    9, 6.65, "ACTIONS — WHAT CAN WE CONTROL?",
    ha="center", fontsize=12, fontweight="bold",
    color="#147D64",
)

ys = [5.25, 3.75, 2.25]

for (state_id, name, people), y in zip(states, ys):
    draw_box(
        0.65, y, 4.7, 0.85,
        f"{state_id} | {name}", people,
        "#E7EEFF", "#4C76C8",
    )

for (action_id, lights, ac), y in zip(actions, ys):
    draw_box(
        6.65, y, 4.7, 0.85,
        action_id, f"{lights}  •  {ac}",
        "#DDF5EC", "#299878",
    )

ax.text(
    6, 1.05,
    "The MDP will learn which action is best for each state.",
    ha="center", fontsize=12, color="#364152",
)

ax.text(
    6, 0.55,
    "Policy and action selection will be added after transitions and rewards.",
    ha="center", fontsize=10, color="#64748B",
)

plt.tight_layout()

file_path = OUTPUT_DIR / "mdp_states_actions.png"
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Visual saved to: {file_path}")
