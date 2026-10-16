"""Phase 1: read-only inspection of the smart_room_mdp project."""
import ast, importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=== FILE TREE ===")
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and not any(x in p.parts for x in (".git", "__pycache__", ".venv", "venv")):
        print(f"{p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")

print("\n=== src/*.py: imports, functions, top-level constants ===")
for f in sorted((ROOT / "src").glob("*.py")):
    print(f"\n--- {f.name} ---")
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except Exception as e:
        print("  could not parse:", e); continue
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            print("  import:", ast.unparse(n))
        elif isinstance(n, ast.FunctionDef):
            print(f"  def {n.name}({ast.unparse(n.args)})")
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
            print("  const:", ast.unparse(n)[:200])
    has_main = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)
    print("  runs code under __main__ guard:", has_main)

print("\n=== data/hmm_predictions.csv ===")
try:
    import pandas as pd, numpy as np
    df = pd.read_csv(ROOT / "data" / "hmm_predictions.csv")
    print("shape:", df.shape); print("columns:", list(df.columns)); print(df.head(3).to_string())
    pc = [c for c in df.columns if c.startswith("P(S")]
    if pc:
        s = df[pc].sum(axis=1)
        print("prob row-sum min/max:", s.min(), s.max(), "| NaNs:", int(df[pc].isna().sum().sum()))
    if "timestamp" in df: print("timestamp sample:", df["timestamp"].iloc[0], "| sorted:", df["timestamp"].is_monotonic_increasing)
    for c in ("predicted_state", "true_state"):
        if c in df: print(c, "values:", sorted(df[c].dropna().unique().tolist()))
except Exception as e:
    print("could not inspect:", e)

print("\n=== output/ ===")
out = ROOT / "output"
print([p.name for p in sorted(out.iterdir())] if out.exists() else "missing")

print("\n=== packages ===")
for m in ("streamlit", "plotly", "pandas", "numpy"):
    spec = importlib.util.find_spec(m)
    print(m, "installed" if spec else "MISSING")
print("python:", sys.version.split()[0])