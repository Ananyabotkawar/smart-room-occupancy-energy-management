"""Smart Room Occupancy Estimation & Energy Management Simulator.

HMM belief  ->  belief-weighted MDP action value  ->  lights / AC decision.
Software simulation only: no real hardware is controlled.
Run:  python -m streamlit run app.py
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:  # launched with `python app.py`
    print("\nThis is a Streamlit app. Start it with:\n\n    python -m streamlit run app.py\n")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
HMM_PATH = ROOT / "data" / "hmm_predictions.csv"

st.set_page_config(page_title="Smart Room Simulator", page_icon="🏢", layout="wide")
st.markdown("""<style>
.card{background:#111a2e;border:1px solid #1e2b47;border-radius:14px;padding:14px 16px;margin-bottom:10px}
.card .t{color:#8fa3c2;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
.card .v{color:#e5edf7;font-size:1.5rem;font-weight:700}
.bar{height:8px;border-radius:6px;background:#1e2b47;margin-top:6px}
.bar>div{height:8px;border-radius:6px;background:#38bdf8}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.8rem;font-weight:600;border:1px solid}
.ok{color:#34d399;border-color:#34d399;background:#34d3991a}
.warn{color:#fbbf24;border-color:#fbbf24;background:#fbbf241a}
.run{color:#38bdf8;border-color:#38bdf8;background:#38bdf81a}
</style>""", unsafe_allow_html=True)

STATE_SHORT = ["S0", "S1", "S2"]
STATE_LABEL = ["S0 · Empty", "S1 · 1 person", "S2 · 2–3 people"]
LABELS = {"S0": 0, "S1": 1, "S2": 2, "0": 0, "1": 1, "2": 2}


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading MDP model and solving...")
def load_model() -> dict:
    """Reuse the existing project modules; nothing is re-implemented here."""
    import mdp_model as m
    from value_iteration import value_iteration
    from policy_iteration import policy_iteration

    def quiet(fn):
        try:
            return fn(verbose=False)
        except TypeError:  # modules not patched with verbose flag yet
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()

    v_vi, pol_vi = quiet(value_iteration)[:2]
    v_pi, pol_pi = quiet(policy_iteration)[:2]
    R, T, g = np.asarray(m.REWARDS, float), np.asarray(m.TRANSITIONS, float), float(m.GAMMA)
    if R.shape != (3, 3) or T.shape != (3, 3):
        raise ValueError("Reward/transition matrices must be 3x3.")
    if not np.allclose(T.sum(axis=1), 1) or (T < 0).any():
        raise ValueError("Transition rows must be probabilities summing to 1.")
    if not 0 <= g < 1:
        raise ValueError("Discount factor must satisfy 0 <= gamma < 1.")
    # Q_sa[s,a] = R[s,a] + gamma * sum_s' P(s'|s) V(s')
    Q_sa = R + g * (T @ v_vi)[:, None]
    return dict(R=R, T=T, gamma=g, V=v_vi, policy=np.asarray(pol_vi, int), Q_sa=Q_sa,
                actions=m.ACTIONS,
                agree=bool(np.allclose(v_vi, v_pi, atol=1e-6) and np.array_equal(pol_vi, pol_pi)))


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading and validating HMM predictions...")
def load_hmm(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Put hmm_predictions.csv in the data/ folder.")
    raw = pd.read_csv(p)
    need = ["timestamp", "P(S0)", "P(S1)", "P(S2)"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError(f"hmm_predictions.csv is missing columns: {miss}")
    notes = []
    df = pd.DataFrame({"timestamp": pd.to_datetime(raw["timestamp"], errors="coerce")})
    for i in range(3):
        df[f"P{i}"] = pd.to_numeric(raw[f"P(S{i})"], errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    P = df[["P0", "P1", "P2"]].to_numpy()
    sums = P.sum(axis=1)
    ok = (np.isfinite(P).all(axis=1) & (P >= 0).all(axis=1) & (np.abs(sums - 1) <= 0.01)
          & df["timestamp"].notna().to_numpy())
    if (~ok).any():
        notes.append(f"Excluded {int((~ok).sum())} invalid record(s) "
                     f"({bad_ts} bad timestamps; others had non-finite, negative or non-normalised probabilities).")
    keep = ok
    for col, name in (("predicted_state", "csv_pred"), ("true_state", "true")):
        if col in raw.columns:
            s = raw[col].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
            df[name] = s.map(LABELS)
            if df[name][keep].isna().any():
                notes.append(f"`{col}` has values outside S0/S1/S2; those rows are not scored.")
        else:
            df[name] = np.nan
            notes.append(f"`{col}` column not found: related evaluation is disabled.")
    df = df[keep].copy()
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp", kind="stable")
        notes.append("Records were not in time order and have been sorted.")
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid records left after validation.")
    if df["timestamp"].duplicated().any():
        notes.append(f"{int(df['timestamp'].duplicated().sum())} duplicate timestamp(s) found.")
    gaps = df["timestamp"].diff().dt.total_seconds().dropna()
    if len(gaps):
        med = gaps.median()
        odd = int((gaps > 1.5 * med).sum())
        if odd:
            notes.append(f"{odd} gap(s) longer than 1.5x the median {med:.0f}s (missing records).")
    df["belief_argmax"] = df[["P0", "P1", "P2"]].to_numpy().argmax(axis=1)
    if df["csv_pred"].notna().all():
        agree = float((df["csv_pred"] == df["belief_argmax"]).mean())
        notes.append(f"CSV predicted_state equals the most probable belief state on {agree:.1%} of rows.")
    return df, notes


# ------------------------------------------------------------------ simulation
def simulate(df, model, p):
    """Vectorised run over all records; playback just reveals rows progressively."""
    B = df[["P0", "P1", "P2"]].to_numpy()
    n = len(df)
    Q = B @ model["Q_sa"]                       # belief-weighted expected action values
    a_ai = Q.argmax(axis=1)
    mps = B.argmax(axis=1)
    baselines = {
        "Always ON (A2)": np.full(n, 2),
        "Most-probable-state policy": model["policy"][mps],
        "Always A1 (lights ON, AC LOW)": np.full(n, 1),
    }
    a_base = baselines[p["baseline"]]
    watts = np.array([0.0, p["light_w"] + p["ac_low_w"], p["light_w"] + p["ac_high_w"]])
    if p["use_gaps"] and n > 1:
        t = df["timestamp"].to_numpy()
        dur = np.append(np.diff(t).astype("timedelta64[ms]").astype(float) / 1000, np.nan)
        dur[-1] = np.nanmedian(dur[:-1])
    else:
        dur = np.full(n, float(p["step_s"]))
    e_ai = watts[a_ai] * dur / 3600
    e_base = watts[a_base] * dur / 3600
    exp_r = (B @ model["R"])[np.arange(n), a_ai]  # belief-weighted immediate reward (no ground truth)
    res = pd.DataFrame({
        "record": np.arange(1, n + 1), "timestamp": df["timestamp"],
        "P(S0)": B[:, 0], "P(S1)": B[:, 1], "P(S2)": B[:, 2],
        "most_probable_state": [STATE_SHORT[i] for i in mps],
        "selected_action": [f"A{i}" for i in a_ai],
        "expected_action_value": Q[np.arange(n), a_ai],
        "expected_reward": exp_r,
        "power_W": watts[a_ai], "step_seconds": dur, "energy_Wh": e_ai,
        "cumulative_energy_Wh": e_ai.cumsum(), "cumulative_reward": exp_r.cumsum(),
        "baseline": p["baseline"], "baseline_action": [f"A{i}" for i in a_base],
        "baseline_energy_Wh": e_base, "baseline_cumulative_Wh": e_base.cumsum(),
        "true_state": [STATE_SHORT[int(i)] if pd.notna(i) else "" for i in df["true"]],
    })
    return res, Q, a_ai, mps


# ------------------------------------------------------------------ room visual
def room_svg(state: int, action: int) -> str:
    """Animated office. Depends only on (state, action), so it is not redrawn while nothing changes."""
    lights = action in (1, 2)
    ac = ["OFF", "LOW", "HIGH"][action]
    acc = {"OFF": "#64748b", "LOW": "#22d3ee", "HIGH": "#34d399"}[ac]
    people = {0: [], 1: [400], 2: [330, 470]}[state]
    s = ['<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:16px">',
         '<defs>'
         '<linearGradient id="wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1c2b4a"/><stop offset="1" stop-color="#142039"/></linearGradient>'
         '<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2b3b5e"/><stop offset="1" stop-color="#1a263f"/></linearGradient>'
         '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3b6aa8"/><stop offset="1" stop-color="#1e3a5f"/></linearGradient>'
         '<radialGradient id="glow" cx="50%" cy="0%" r="100%"><stop offset="0" stop-color="#fde68a" stop-opacity=".5"/><stop offset="1" stop-color="#fde68a" stop-opacity="0"/></radialGradient>'
         '</defs>',
         '<rect width="800" height="450" fill="#0b1220"/>',
         '<rect x="0" y="0" width="800" height="270" fill="url(#wall)"/>',
         '<rect x="0" y="236" width="800" height="34" fill="#18263f"/>',
         '<rect x="0" y="270" width="800" height="150" fill="url(#floor)"/>']
    for k in range(1, 7):
        s.append(f'<line x1="0" y1="{270 + k * 22}" x2="800" y2="{270 + k * 22}" stroke="#1a263f" stroke-width="1"/>')
    # window with curtains and sunlight sky
    s.append('<rect x="40" y="52" width="190" height="130" fill="url(#sky)" stroke="#3d5a85" stroke-width="5"/>'
             '<line x1="135" y1="52" x2="135" y2="182" stroke="#3d5a85" stroke-width="4"/>'
             '<line x1="40" y1="117" x2="230" y2="117" stroke="#3d5a85" stroke-width="4"/>'
             '<rect x="28" y="46" width="26" height="146" rx="6" fill="#4c1d95" opacity=".75"/>'
             '<rect x="216" y="46" width="26" height="146" rx="6" fill="#4c1d95" opacity=".75"/>')
    # whiteboard, plant, rug
    s.append('<rect x="300" y="62" width="200" height="110" rx="6" fill="#e2e8f0" opacity=".9"/>'
             '<line x1="318" y1="90" x2="440" y2="90" stroke="#94a3b8" stroke-width="3"/>'
             '<line x1="318" y1="112" x2="470" y2="112" stroke="#94a3b8" stroke-width="3"/>'
             '<line x1="318" y1="134" x2="410" y2="134" stroke="#94a3b8" stroke-width="3"/>'
             '<rect x="705" y="320" width="46" height="46" rx="6" fill="#92400e"/>'
             '<ellipse cx="716" cy="300" rx="12" ry="30" fill="#16a34a" transform="rotate(-22 716 300)"/>'
             '<ellipse cx="740" cy="298" rx="12" ry="32" fill="#22c55e" transform="rotate(20 740 298)"/>'
             '<ellipse cx="728" cy="290" rx="11" ry="34" fill="#15803d"/>'
             '<ellipse cx="400" cy="392" rx="270" ry="30" fill="#22304f" opacity=".85"/>')
    if lights:
        s.append('<polygon points="250,24 350,24 430,300 170,300" fill="url(#glow)"/>'
                 '<polygon points="450,24 550,24 630,300 370,300" fill="url(#glow)"/>')
    # chairs, then people (behind the desk)
    for cx in (330, 400, 470):
        s.append(f'<rect x="{cx - 28}" y="252" width="56" height="62" rx="12" fill="#334155"/>')
    for i, x in enumerate(people):
        s.append(f'<g><animateTransform attributeName="transform" type="translate" values="0 0;0 -2.5;0 0" '
                 f'dur="{2.2 + i * 0.5}s" repeatCount="indefinite"/>'
                 f'<rect x="{x - 24}" y="266" width="48" height="52" rx="18" fill="#38bdf8"/>'
                 f'<circle cx="{x}" cy="246" r="18" fill="#f1c9a5"/>'
                 f'<path d="M{x - 18},244 A18,18 0 0 1 {x + 18},244 Z" fill="#1e293b"/>'
                 f'<line x1="{x - 20}" y1="280" x2="{x - 12}" y2="304" stroke="#38bdf8" stroke-width="9" stroke-linecap="round"/>'
                 f'<line x1="{x + 20}" y1="280" x2="{x + 12}" y2="304" stroke="#38bdf8" stroke-width="9" stroke-linecap="round"/></g>')
    s.append('<rect x="240" y="304" width="320" height="16" rx="6" fill="#8b6b47"/>'
             '<rect x="250" y="320" width="300" height="8" fill="#6b4f33"/>'
             '<rect x="262" y="328" width="10" height="58" fill="#6b4f33"/>'
             '<rect x="528" y="328" width="10" height="58" fill="#6b4f33"/>')
    if not lights:
        s.append('<rect width="800" height="408" fill="#050b18" opacity=".62"/>')
    for x in people:  # laptops glow even with lights off
        s.append(f'<rect x="{x - 22}" y="290" width="44" height="14" rx="2" fill="#0f172a"/>'
                 f'<rect x="{x - 19}" y="292" width="38" height="10" rx="1" fill="#7dd3fc" opacity=".9"/>')
    fx = "#fde68a" if lights else "#3b4863"
    s.append(f'<rect x="250" y="12" width="100" height="12" rx="5" fill="{fx}"/>'
             f'<rect x="450" y="12" width="100" height="12" rx="5" fill="{fx}"/>')
    # AC unit + flowing air
    s.append(f'<rect x="570" y="28" width="200" height="50" rx="10" fill="#1e293b" stroke="{acc}" stroke-width="3"/>'
             f'<rect x="582" y="62" width="176" height="7" rx="3" fill="{acc}" opacity=".7"/>'
             f'<text x="596" y="52" font-size="15" font-weight="700" fill="{acc}">AC {ac}</text>'
             f'<circle cx="748" cy="44" r="6" fill="{acc}"/>')
    n_air, dur = {"OFF": (0, 1), "LOW": (3, 1.3), "HIGH": (6, 0.55)}[ac]
    for k in range(n_air):
        x = 592 + k * 30
        s.append(f'<path d="M{x},84 q9,18 0,36 t0,36 t0,36" fill="none" stroke="{acc}" stroke-width="3" '
                 f'stroke-linecap="round" stroke-dasharray="7 11" opacity=".85">'
                 f'<animate attributeName="stroke-dashoffset" from="36" to="0" dur="{dur}s" repeatCount="indefinite"/></path>')
    def badge(x, w, txt, fill):
        return (f'<rect x="{x}" y="414" width="{w}" height="28" rx="14" fill="{fill}"/>'
                f'<text x="{x + w / 2}" y="433" text-anchor="middle" font-size="14" font-weight="600" fill="#fff">{txt}</text>')
    s.append(badge(24, 150, "Lights ON" if lights else "Lights OFF", "#059669" if lights else "#475569"))
    s.append(badge(186, 150, f"AC {ac}", "#059669" if ac != "OFF" else "#475569"))
    s.append(badge(348, 300, f"Estimated: {STATE_LABEL[state]}", "#1d4ed8"))
    s.append(badge(660, 116, "SIMULATION", "#334155"))
    s.append("</svg>")
    return "".join(s)


def dark(fig, title, x, y, h=250):
    fig.update_layout(template="plotly_dark", title=dict(text=title, font=dict(size=14)), xaxis_title=x,
                      yaxis_title=y, height=h, margin=dict(l=8, r=8, t=40, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.3), showlegend=True)
    return fig


def thin(d, cap=300):
    return d.iloc[::max(1, len(d) // cap)]


def now_line(fig, t):
    fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper", line=dict(color="#f8fafc", width=1, dash="dot"))


# ------------------------------------------------------------------ main
def main():
    ss = st.session_state
    ss.setdefault("idx", 0)
    ss.setdefault("playing", False)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"MDP model failed to load: {type(e).__name__}: {e}. Check src/mdp_model.py, "
                 "src/value_iteration.py, src/policy_iteration.py and data/Occupancy_Estimation.csv.")
        st.stop()
    try:
        df_all, notes = load_hmm(str(HMM_PATH))
    except Exception as e:
        st.error(f"HMM data problem: {e}")
        st.stop()

    sb = st.sidebar
    sb.header("Run settings")
    chg = np.flatnonzero(np.diff(df_all["belief_argmax"].to_numpy()) != 0) + 1
    opts = ["Full dataset", "First 500"]
    if len(chg):
        opts.insert(0, "Demo window around first state change")
    choice = sb.selectbox("Records to simulate", opts, index=0)
    if choice.startswith("Demo"):
        lo, hi = max(0, int(chg[0]) - 100), min(len(df_all), int(chg[0]) + 150)
    elif choice == "First 500":
        lo, hi = 0, min(500, len(df_all))
    else:
        lo, hi = 0, len(df_all)
    speed = sb.select_slider("Playback speed (records per tick)", [1, 2, 3, 5, 10, 20, 50], value=3)
    sb.header("Energy assumptions (illustrative)")
    sb.caption("Not measured data. Edit to match your own assumptions.")
    light_w = sb.number_input("Lighting power when ON (W)", 0.0, 5000.0, 60.0, 5.0)
    ac_low = sb.number_input("AC power, LOW (W)", 0.0, 10000.0, 800.0, 50.0)
    ac_high = sb.number_input("AC power, HIGH (W)", 0.0, 10000.0, 1500.0, 50.0)
    use_gaps = sb.checkbox("Use real timestamp gaps as step duration", True)
    step_s = sb.number_input("Fixed step duration (s)", 1.0, 3600.0, 30.0, 1.0, disabled=use_gaps)
    tariff = sb.number_input("Tariff (currency units per kWh)", 0.0, 1000.0, 8.0, 0.5)
    baseline = sb.selectbox("Baseline controller", ["Always ON (A2)", "Most-probable-state policy",
                                                    "Always A1 (lights ON, AC LOW)"])

    df = df_all.iloc[lo:hi].reset_index(drop=True)
    if ss.get("window") != (lo, hi):
        ss.window, ss.idx, ss.playing = (lo, hi), 0, False
    params = dict(light_w=light_w, ac_low_w=ac_low, ac_high_w=ac_high, use_gaps=use_gaps,
                  step_s=step_s, baseline=baseline)
    res, Q, a_ai, mps = simulate(df, model, params)
    n = len(res)
    ss.idx = int(min(max(ss.idx, 0), n - 1))

    def toggle():
        if ss.playing:
            ss.playing = False
        else:
            if ss.idx >= n - 1:
                ss.idx = 0
            ss.playing = True
    def nxt():  ss.playing = False; ss.idx = min(ss.idx + 1, n - 1)
    def prv():  ss.playing = False; ss.idx = max(ss.idx - 1, 0)
    def rst():  ss.playing = False; ss.idx = 0
    def seek(): ss.playing = False; ss.idx = int(ss.seek) - 1

    # ---- header + controls (full-page rerun only when these are used)
    h1, h2 = st.columns([5, 1])
    h1.markdown("### 🏢 Smart Room Simulator &nbsp;·&nbsp; HMM belief → MDP decision → lights & AC")
    cls, label = ("run", "● LIVE") if ss.playing else ("ok", "Ready") if ss.idx == 0 else ("warn", "Paused")
    h2.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)

    c = st.columns([1.3, 1, 1, 1, 4])
    c[0].button("⏸ Pause" if ss.playing else ("▶ Start" if ss.idx == 0 else "▶ Resume"),
                on_click=toggle, type="primary", use_container_width=True)
    c[1].button("◀ Prev", on_click=prv, use_container_width=True)
    c[2].button("Next ▶", on_click=nxt, use_container_width=True)
    c[3].button("↺ Reset", on_click=rst, use_container_width=True)
    if not ss.playing and n > 1:
        ss.seek = ss.idx + 1
        c[4].slider("Jump to record", 1, n, key="seek", on_change=seek)
    elif ss.playing:
        c[4].caption("Playing… press Pause to scrub or inspect details.")

    # ---- live view: only this fragment re-runs on each tick
    @st.fragment(run_every=0.35 if ss.playing else None)
    def live():
        if ss.playing:
            if ss.idx + speed >= n - 1:
                ss.idx, ss.playing = n - 1, False
                st.rerun(scope="app")
            ss.idx += speed
        idx = int(ss.idx)
        row = res.iloc[idx]
        b = np.array([row["P(S0)"], row["P(S1)"], row["P(S2)"]])
        a, s_hat, q = int(a_ai[idx]), int(mps[idx]), Q[idx]

        st.progress((idx + 1) / n)
        st.caption(f"Record {idx + 1} of {n} · {row['timestamp']} · {(idx + 1) / n:.0%} complete")

        left, right = st.columns([3, 2])
        with left:
            st.markdown(room_svg(s_hat, a), unsafe_allow_html=True)
            st.caption("Illustration of the *estimated* state (S2 = 2–3 people, two drawn). "
                       "Software simulation; no real hardware or people are measured.")
        with right:
            for i in range(3):
                st.markdown(
                    f'<div class="card" style="padding:8px 14px;margin-bottom:6px">'
                    f'<div class="t">P({STATE_SHORT[i]}) · {STATE_LABEL[i].split("· ")[1]}</div>'
                    f'<div class="v" style="font-size:1.25rem">{b[i]:.1%}</div>'
                    f'<div class="bar"><div style="width:{b[i] * 100:.1f}%"></div></div></div>',
                    unsafe_allow_html=True)
            st.success(f"Decision: {model['actions'][a]}")
            st.dataframe(pd.DataFrame({"Action": [model["actions"][i] for i in range(3)],
                                       "Q(b,a)": np.round(q, 2)}), hide_index=True)
            order = np.argsort(-q)
            why = (f"Top belief: {STATE_SHORT[s_hat]} ({b[s_hat]:.1%}). Best expected action value: "
                   f"{model['actions'][order[0]]} (Q={q[order[0]]:.2f}), runner-up "
                   f"{model['actions'][order[1]]} (Q={q[order[1]]:.2f}).")
            if order[0] != model["policy"][s_hat]:
                why += " Differs from the most-probable-state policy because the belief is split."
            st.caption(why)

        r = res.iloc[: idx + 1]
        e_ai, e_base = r["energy_Wh"].sum(), r["baseline_energy_Wh"].sum()
        saving = e_base - e_ai
        pct = f"{saving / e_base * 100:.1f}%" if e_base > 0 else "n/a"
        switches = int((np.diff(a_ai[: idx + 1]) != 0).sum())
        m = st.columns(5)
        m[0].metric("AI energy", f"{e_ai:.1f} Wh", f"{e_ai / 1000 * tariff:.3f} cost units", delta_color="off")
        m[1].metric("Baseline energy", f"{e_base:.1f} Wh", baseline, delta_color="off")
        m[2].metric("Saving", f"{saving:.1f} Wh", pct)
        m[3].metric("Cumulative reward", f"{r['expected_reward'].sum():.1f}", delta_color="off")
        m[4].metric("Action switches", switches)

        rt, t_now = thin(r), row["timestamp"]
        x_rng = [res["timestamp"].iloc[0], res["timestamp"].iloc[-1]]
        g = st.columns(3)
        f = go.Figure()
        for col in ("P(S0)", "P(S1)", "P(S2)"):
            f.add_scatter(x=rt["timestamp"], y=rt[col], name=col, mode="lines")
        f.update_xaxes(range=x_rng); f.update_yaxes(range=[0, 1.02]); now_line(f, t_now)
        g[0].plotly_chart(dark(f, "HMM belief", "Time", "Probability"))
        f = go.Figure(go.Scatter(x=rt["timestamp"], y=[int(x[1]) for x in rt["selected_action"]],
                                 line_shape="hv", name="Action", line=dict(color="#34d399")))
        f.update_xaxes(range=x_rng); f.update_yaxes(tickvals=[0, 1, 2], ticktext=["A0", "A1", "A2"], range=[-0.2, 2.2])
        now_line(f, t_now)
        g[1].plotly_chart(dark(f, "Selected action", "Time", "Action"))
        f = go.Figure()
        f.add_scatter(x=rt["timestamp"], y=rt["cumulative_energy_Wh"], name="AI controller", line=dict(color="#34d399"))
        f.add_scatter(x=rt["timestamp"], y=rt["baseline_cumulative_Wh"], name="Baseline", line=dict(color="#fbbf24"))
        f.update_xaxes(range=x_rng); now_line(f, t_now)
        g[2].plotly_chart(dark(f, "Cumulative energy (simulated)", "Time", "Wh"))
        st.caption("Energy values come from the editable sidebar assumptions, not measurements. "
                   "AI controller and baseline use the same records and durations.")

    live()

    # ---- details (refresh when paused / on any full rerun)
    idx = int(ss.idx)
    r = res.iloc[: idx + 1]
    with st.expander("Analytics, log and export (updates when paused)"):
        t_all = df["true"].iloc[: idx + 1]
        valid = t_all.notna()
        if valid.any():
            acc = float((mps[: idx + 1][valid.to_numpy()] == t_all[valid].astype(int).to_numpy()).mean())
            maj = float(t_all[valid].astype(int).value_counts(normalize=True).max())
            st.write(f"**State accuracy so far:** {acc:.1%} (always predicting the commonest state: {maj:.1%}). "
                     "`true_state` is used for evaluation only, never for control.")
        else:
            st.write("State accuracy unavailable: no valid `true_state` labels.")
        c1, c2 = st.columns(2)
        cnt = r["selected_action"].value_counts().reindex(["A0", "A1", "A2"], fill_value=0)
        c1.plotly_chart(dark(go.Figure(go.Bar(x=cnt.index, y=cnt.values)), "Action distribution", "Action", "Records"))
        c2.plotly_chart(dark(go.Figure(go.Bar(x=STATE_SHORT, y=model["V"])), "State values V(s)", "State", "Value"))
        show = ["record", "timestamp", "P(S0)", "P(S1)", "P(S2)", "most_probable_state", "selected_action",
                "expected_action_value", "power_W", "energy_Wh", "cumulative_energy_Wh", "cumulative_reward", "true_state"]
        tbl = r[show].tail(10).iloc[::-1].copy()
        for col in tbl.select_dtypes("float").columns:
            tbl[col] = tbl[col].round(4)
        st.dataframe(tbl, hide_index=True)
        st.download_button(f"Download results CSV (records 1–{idx + 1} of this run)", r.to_csv(index=False).encode(),
                           "smart_room_simulation_results.csv", "text/csv")
    with st.expander("Data validation and model notes"):
        st.write(f"{len(df_all)} valid records loaded; simulating records {lo + 1}–{hi} ({n}). Metrics cover this window.")
        for note in notes:
            st.write("• " + note)
        st.write("• Value Iteration and Policy Iteration " + ("agree." if model["agree"] else "DISAGREE."))
        st.markdown("""
- **Learned from data:** HMM beliefs (existing output) and the transition matrix (observed occupancy changes).
- **Illustrative assumptions:** reward matrix, discount factor, all power and tariff values.
- **Controller:** Q(b,a) = Σₛ b(s)[R(s,a) + γ Σₛ' P(s'|s) V(s')], an approximation using HMM beliefs and MDP values, **not** a solved POMDP.
- **Transitions are identical for every action**, so the choice is driven by belief-weighted rewards. This does not show that lighting or AC affects occupancy.
- **Dataset:** S1 (one person) does not occur, so the controller mostly switches between A0 and A2.
- **Bayesian Network:** not implemented (future integration point). `true_state` is evaluation-only.
""")


main()
