

import json, glob, os, re
import numpy as np
from scipy import stats
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wyniki_final")

# 1. Wczytaj dane 
groups = defaultdict(list)

for trait in sorted(os.listdir(BASE)):
    files = glob.glob(os.path.join(BASE, trait, "*.json"))
    for f in sorted(files):
        fname = os.path.basename(f)
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)

        m = re.match(r'^([A-Z])_(\w+)-([A-Z])_(\w+)_', fname)
        if not m:
            continue
        pole1 = m.group(2)          # "wysoki" lub "niski"
        order = ("high_first" if pole1 == "wysoki" else "low_first")
        key   = f"{trait}_{order}"

        agents     = [a["name"] for a in d["meta"]["agents"]]
        agent_high = agents[0] if pole1 == "wysoki" else agents[1]
        agent_low  = agents[1] if pole1 == "wysoki" else agents[0]


        consensus  = d["decision"].get("consensus_reached", False)
        delta_cos  = d["metrics"]["between_agent_similarity"].get("delta")
        shift_mean = d["metrics"]["opinion_shift"].get("mean")
        hedge_mean = d["metrics_j"]["hedging_rate"].get("overall")
        flips_mean = d["metrics"]["answer_flip_rate"]["summary"].get("mean_nof")

 
        shift_pa = d["metrics"]["opinion_shift"].get("per_agent", {})
        hedge_pa = d["metrics_j"]["hedging_rate"].get("per_agent", {})
        flip_pa  = d["metrics"]["answer_flip_rate"].get("per_agent", {})
        rep_pa   = d["metrics"]["history_repetition"].get("per_agent", {})

        groups[key].append({
            # Tabela 1
            "consensus":  consensus,
            "delta_cos":  delta_cos,
            "shift_mean": shift_mean,
            "hedge_mean": hedge_mean,
            "flips_mean": flips_mean,
            "shift_h": shift_pa.get(agent_high),
            "hedge_h": hedge_pa.get(agent_high, {}).get("hedging_rate"),
            "flips_h": flip_pa.get(agent_high,  {}).get("nof"),
            "rep_h":   rep_pa.get(agent_high,   {}).get("repetition_rate"),
            "shift_l": shift_pa.get(agent_low),
            "hedge_l": hedge_pa.get(agent_low,  {}).get("hedging_rate"),
            "flips_l": flip_pa.get(agent_low,   {}).get("nof"),
            "rep_l":   rep_pa.get(agent_low,    {}).get("repetition_rate"),
        })

# ── 2. Pomocnicze ─────────────────────────────────────────────────────────

def nanmean(lst):
    vals = [x for x in lst if x is not None]
    return np.mean(vals) if vals else float("nan")

def paired_p(a, b):
    """Paired t-test; obcina do wspólnej długości."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    _, p = stats.ttest_rel(a[:n], b[:n])
    return p

def fmt_p(p):
    if np.isnan(p):
        return "  nan"
    s = f"{p:.2f}"
    return s + ("*" if p < 0.05 else " ")

# 3. Tabela 1 
print("=" * 75)
print("Tabela 1: Dynamika debaty per para cech")
print("=" * 75)
print(f"{'Para':<35} {'N':>4} {'Cons':>6} {'ΔCos':>7} {'Shift':>7} "
      f"{'Hedge':>7} {'Flips':>7}")
print("-" * 75)
for key in sorted(groups.keys()):
    rows = groups[key]
    n    = len(rows)
    cons  = nanmean([r["consensus"]  for r in rows])
    dcos  = nanmean([r["delta_cos"]  for r in rows])
    shift = nanmean([r["shift_mean"] for r in rows])
    hedge = nanmean([r["hedge_mean"] for r in rows])
    flips = nanmean([r["flips_mean"] for r in rows])
    print(f"{key:<35} {n:>4} {cons:>6.3f} {dcos:>7.3f} {shift:>7.3f} "
          f"{hedge:>7.3f} {flips:>7.3f}")

# 4. Tabela 2 

print()
print("=" * 65)
print("Tabela 2: p-wartości paired t-test (wysoki vs niski biegun)")
print("=" * 65)
print(f"{'Para':<35} {'Shift':>8} {'Hedge':>8} {'Flips':>8}")
print("-" * 65)
for key in sorted(groups.keys()):
    rows = groups[key]
    ps = paired_p([r["shift_h"] for r in rows], [r["shift_l"] for r in rows])
    ph = paired_p([r["hedge_h"] for r in rows], [r["hedge_l"] for r in rows])
    pf = paired_p([r["flips_h"] for r in rows], [r["flips_l"] for r in rows])
    print(f"{key:<35} {fmt_p(ps):>8} {fmt_p(ph):>8} {fmt_p(pf):>8}")

# 5. Tabela 3 

print()
print("=" * 70)
print("Tabela 3: Wskaźniki stabilności (Flips i Repetition per biegun)")
print("=" * 70)
print(f"{'Para':<35} {'FlipsH':>7} {'FlipsL':>7} {'RepH':>7} {'RepL':>7}")
print("-" * 70)
for key in sorted(groups.keys()):
    rows = groups[key]
    fh = nanmean([r["flips_h"] for r in rows])
    fl = nanmean([r["flips_l"] for r in rows])
    rh = nanmean([r["rep_h"]   for r in rows])
    rl = nanmean([r["rep_l"]   for r in rows])
    print(f"{key:<35} {fh:>7.2f} {fl:>7.2f} {rh:>7.3f} {rl:>7.3f}")

print()
print("Gotowe.")
