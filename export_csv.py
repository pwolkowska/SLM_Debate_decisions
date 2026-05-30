"""
export_csv.py — Eksportuje wyniki eksperymentów z .json do .csv.

Użycie:
    python export_csv.py                        # skanuje wyniki/ rekurencyjnie
    python export_csv.py --folder wyniki/trojki # tylko wybrany podfolder
    python export_csv.py --output moje.csv      # własna nazwa pliku wyjściowego

Każdy wiersz = jeden plik .json (jedno uruchomienie debaty).

Kolumny identyfikacyjne (z nazwy pliku i meta):
    file, para, proba, seed, wersja,
    agent_1, agent_2, [agent_3],
    num_rounds, architecture, protocol, timestamp

Kolumny metryk globalnych (per agent):
    <agent>_tokens_mean, <agent>_tokens_total,
    <agent>_opinion_shift, <agent>_lexical_ttr,
    <agent>_argument_novelty_mean,
    <agent>_hedging_rate, <agent>_assertiveness_rate, <agent>_net_assertiveness,
    <agent>_claim_grounding_rate,
    <agent>_question_rate,
    <agent>_direct_address_rate, <agent>_interaction_rate, <agent>_combined_address_rate,
    <agent>_rebuttal_sim,
    <agent>_nof, <agent>_tof,
    <agent>_total_opinion_drift,
    <agent>_turn_length_entropy,
    <agent>_history_repetition_rate,

Kolumny metryk globalnych (całość debaty):
    between_agent_sim_initial, between_agent_sim_final, between_agent_sim_delta,
    semantic_diversity_overall,
    redundancy_ratio_mean,
    gini_speaking_time,
    dominant_agent,
    reciprocal_shift_mean_abs_r,
    opinion_traj_monotonicity_overall,
    consensus_reached, convergence_round,
    conv_speed_reached, conv_speed_round, conv_speed_total_delta,

Kolumny per runda (poziomo):
    r<N>_between_sim,
    r<N>_semantic_diversity,
    r<N>_redundancy_ratio,
    r<N>_conv_speed_delta,
    r<N>_question_rate,
    r<N>_agreement_ratio,       (z protokołu consensus)
    r<N>_<agent>_tokens,
    r<N>_<agent>_sentiment,
    r<N>_<agent>_novelty,
    r<N>_<agent>_rebuttal_sim,
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

WYNIKI_FOLDER_PATH = "wyniki_eksperyment_1"
# ---------------------------------------------------------------------------
# Parsowanie nazwy pliku
# ---------------------------------------------------------------------------

FILE_RE = re.compile(
    r"^(?P<para>.+?)_(?P<proba>\d+)_seed(?P<seed>\d+)(?:_v(?P<wersja>\d+))?$"
)

def parse_filename(stem: str) -> dict:
    """Wyciąga para / proba / seed / wersja z nazwy pliku (bez rozszerzenia)."""
    m = FILE_RE.match(stem)
    if m:
        return {
            "para":   m.group("para"),
            "proba":  int(m.group("proba")),
            "seed":   int(m.group("seed")),
            "wersja": int(m.group("wersja")) if m.group("wersja") else 1,
        }
    # fallback gdy format niestandardowy
    return {"para": stem, "proba": None, "seed": None, "wersja": None}


# ---------------------------------------------------------------------------
# Pomocniki
# ---------------------------------------------------------------------------

def _safe(d, *keys, default=None):
    """Bezpieczny dostęp do zagnieżdżonych kluczy."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def _rounds(data: dict) -> list[int]:
    """Zwraca posortowaną listę numerów rund z debate_log."""
    return sorted({e["round"] for e in data.get("debate", [])})


def _agents(data: dict) -> list[str]:
    """Zwraca listę nazw agentów w kolejności wystąpienia."""
    seen, result = set(), []
    for e in data.get("debate", []):
        if e["agent"] not in seen:
            seen.add(e["agent"])
            result.append(e["agent"])
    return result


# ---------------------------------------------------------------------------
# Budowanie wiersza CSV
# ---------------------------------------------------------------------------

def build_row(path: Path, data: dict) -> dict:
    row = {}

    # ── Identyfikatory ──────────────────────────────────────────────
    parsed = parse_filename(path.stem)
    row["file"]   = path.name
    row["para"]   = parsed["para"]
    row["proba"]  = parsed["proba"]
    row["seed"]   = parsed["seed"]
    row["wersja"] = parsed["wersja"]

    meta = data.get("meta", {})
    agent_names = _agents(data)
    rounds      = _rounds(data)

    for i, name in enumerate(agent_names, 1):
        row[f"agent_{i}"] = name

    row["num_rounds"]   = meta.get("num_rounds")
    row["architecture"] = meta.get("architecture")
    row["protocol"]     = meta.get("protocol")
    row["timestamp"]    = meta.get("timestamp")
    row["model"]        = meta.get("model")
    row["seed_meta"]    = _safe(meta, "generation", "seed")

    # ── metrics (z metrics.py) ──────────────────────────────────────
    m  = data.get("metrics", {})
    mj = data.get("metrics_j", {})

    for i, ag in enumerate(agent_names, 1):
        p = f"agent{i}"

        # tokens
        row[f"{p}_tokens_mean"]  = _safe(m, "tokens_per_turn", "per_agent", ag, "mean")
        row[f"{p}_tokens_total"] = _safe(m, "tokens_per_turn", "per_agent", ag, "total")

        # opinion_shift
        row[f"{p}_opinion_shift"] = _safe(m, "opinion_shift", "per_agent", ag)

        # lexical richness
        row[f"{p}_lexical_ttr"] = _safe(m, "lexical_richness", "per_agent", ag, "ttr")

        # answer_flip_rate
        row[f"{p}_nof"] = _safe(m, "answer_flip_rate", "per_agent", ag, "nof")
        row[f"{p}_tof"] = _safe(m, "answer_flip_rate", "per_agent", ag, "tof")

        # history_repetition
        row[f"{p}_history_rep_rate"] = _safe(
            m, "history_repetition", "per_agent", ag, "repetition_rate"
        )

    # between_agent_similarity
    row["between_agent_sim_initial"] = _safe(m, "between_agent_similarity", "initial")
    row["between_agent_sim_final"]   = _safe(m, "between_agent_similarity", "final")
    row["between_agent_sim_delta"]   = _safe(m, "between_agent_similarity", "delta")

    # semantic_diversity
    row["semantic_diversity_overall"] = _safe(m, "semantic_diversity", "overall")

    # redundancy_ratio
    row["redundancy_ratio_mean"] = _safe(m, "redundancy_ratio", "mean")

    # convergence (protokół)
    dec = data.get("decision", {})
    row["consensus_reached"]  = _safe(dec, "consensus_reached")
    row["convergence_round"]  = _safe(dec, "convergence_round")

    # ── metrics_j ───────────────────────────────────────────────────

    # gini
    row["gini_speaking_time"] = _safe(mj, "gini_speaking_time", "gini")
    row["dominant_agent"]     = _safe(mj, "gini_speaking_time", "dominant_agent")

    # reciprocal_shift
    row["reciprocal_shift_mean_abs_r"] = _safe(mj, "reciprocal_shift_index", "mean_abs_r")

    # opinion_trajectory_monotonicity
    row["opinion_traj_mono_overall"] = _safe(mj, "opinion_trajectory_monotonicity", "overall_abs")

    # convergence_speed (relatywna)
    row["conv_speed_reached"]     = _safe(mj, "convergence_speed", "reached")
    row["conv_speed_round"]       = _safe(mj, "convergence_speed", "convergence_round")
    row["conv_speed_total_delta"] = _safe(mj, "convergence_speed", "total_delta")

    for i, ag in enumerate(agent_names, 1):
        p = f"agent{i}"

        # argument_novelty
        row[f"{p}_argument_novelty_mean"] = _safe(mj, "argument_novelty", "per_agent", ag, "mean")

        # hedging / assertiveness
        row[f"{p}_hedging_rate"]      = _safe(mj, "hedging_rate",       "per_agent", ag, "hedging_rate")
        row[f"{p}_assertiveness_rate"]= _safe(mj, "assertiveness_score","per_agent", ag, "assertiveness_rate")
        row[f"{p}_net_assertiveness"] = _safe(mj, "assertiveness_score","per_agent", ag, "net_assertiveness")

        # claim_grounding
        row[f"{p}_claim_grounding_rate"] = _safe(mj, "claim_grounding", "per_agent", ag, "grounding_rate")

        # question_density
        row[f"{p}_question_rate"] = _safe(mj, "question_density", "per_agent", ag, "question_rate")

        # direct_address
        row[f"{p}_direct_address_rate"]   = _safe(mj, "direct_address_rate", "per_agent", ag, "address_rate")
        row[f"{p}_interaction_rate"]      = _safe(mj, "direct_address_rate", "per_agent", ag, "interaction_rate")
        row[f"{p}_combined_address_rate"] = _safe(mj, "direct_address_rate", "per_agent", ag, "combined_rate")

        # rebuttal_depth
        row[f"{p}_rebuttal_sim"] = _safe(mj, "rebuttal_depth", "per_agent", ag, "mean_rebuttal_sim")

        # total_opinion_drift
        row[f"{p}_total_opinion_drift"] = _safe(mj, "total_opinion_drift", "per_agent", ag, "drift")

        # turn_length_entropy
        row[f"{p}_turn_length_entropy"] = _safe(mj, "turn_length_entropy", "per_agent", ag, "entropy_bits")

    # ── Per runda (poziomo) ─────────────────────────────────────────

    # Indeksy per runda z between_agent_similarity
    # JSON serializuje int-klucze jako stringi — konwertujemy z powrotem na int
    def _int_keys(d: dict) -> dict:
        return {int(k): v for k, v in d.items()}

    bas_per_round = _int_keys(_safe(m, "between_agent_similarity", "per_round") or {})
    sd_per_round  = _int_keys(_safe(m, "semantic_diversity",       "per_round") or {})
    rr_per_round  = _int_keys(_safe(m, "redundancy_ratio",         "per_round") or {})
    cs_delta      = _int_keys(_safe(mj, "convergence_speed", "delta_per_round") or {})
    qd_per_round  = _int_keys(_safe(mj, "question_density",        "per_round") or {})

    # agreement_ratio z protokołu consensus per runda
    consensus_rounds = {
        r["round"]: r.get("agreement_ratio")
        for r in dec.get("consensus_rounds", [])
    }

    # Per agent: sentiment trajectory
    sentiment_traj: dict[str, dict] = {}
    for ag in agent_names:
        traj = _safe(m, "opinion_shift_sentiment", "per_agent", ag, "trajectory") or []
        sentiment_traj[ag] = {t["round"]: t.get("sentiment") for t in traj}

    # Per agent: novelty per turn
    novelty_traj: dict[str, dict] = {}
    for ag in agent_names:
        turns = _safe(mj, "argument_novelty", "per_agent", ag, "per_turn") or []
        novelty_traj[ag] = {t["round"]: t.get("novelty") for t in turns}

    # Per agent: tokens per runda (z debate_log)
    tokens_by_round_agent: dict[int, dict[str, int]] = {}
    for e in data.get("debate", []):
        tokens_by_round_agent.setdefault(e["round"], {})[e["agent"]] = e["tokens"]

    # Per entry: rebuttal_sim per (runda, agent)
    rebuttal_by_round_agent: dict[int, dict[str, float]] = {}
    for entry in _safe(mj, "rebuttal_depth", "per_entry") or []:
        rnd = entry.get("round")
        ag  = entry.get("agent")
        sim = entry.get("rebuttal_sim")
        if rnd is not None and ag is not None and sim is not None:
            rebuttal_by_round_agent.setdefault(rnd, {})[ag] = sim

    for rnd in rounds:
        r = str(rnd)

        row[f"r{r}_between_sim"]        = bas_per_round.get(rnd)
        row[f"r{r}_semantic_diversity"] = sd_per_round.get(rnd)
        row[f"r{r}_redundancy_ratio"]   = rr_per_round.get(rnd)
        row[f"r{r}_conv_speed_delta"]   = cs_delta.get(rnd)
        row[f"r{r}_question_rate"]      = qd_per_round.get(rnd)
        row[f"r{r}_agreement_ratio"]    = consensus_rounds.get(rnd)

        for i, ag in enumerate(agent_names, 1):
            p = f"agent{i}"
            row[f"r{r}_{p}_tokens"]      = tokens_by_round_agent.get(rnd, {}).get(ag)
            row[f"r{r}_{p}_sentiment"]   = sentiment_traj.get(ag, {}).get(rnd)
            row[f"r{r}_{p}_novelty"]     = novelty_traj.get(ag, {}).get(rnd)
            row[f"r{r}_{p}_rebuttal_sim"]= rebuttal_by_round_agent.get(rnd, {}).get(ag)

    return row


# ---------------------------------------------------------------------------
# Zbieranie plików i zapis CSV
# ---------------------------------------------------------------------------

def collect_jsons(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.json"))


def export(folder: Path, output: Path):
    files = collect_jsons(folder)
    if not files:
        print(f"Brak plików .json w {folder}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rows.append(build_row(f, data))
        except Exception as e:
            print(f"  POMINIĘTO {f.name}: {e}", file=sys.stderr)

    if not rows:
        print("Żaden plik nie został wczytany.", file=sys.stderr)
        sys.exit(1)

    # Unia wszystkich kluczy z zachowaniem kolejności
    all_keys: list[str] = []
    seen_keys: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Zapisano {len(rows)} wierszy → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eksportuje wyniki .json z eksperymentów do .csv"
    )
    parser.add_argument(
        "--folder", type=Path, default=Path(WYNIKI_FOLDER_PATH),
        help="Folder z plikami .json (domyślnie: wyniki/)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("wyniki_export.csv"),
        help="Nazwa pliku wyjściowego (domyślnie: wyniki_export.csv)"
    )
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Folder nie istnieje: {args.folder}", file=sys.stderr)
        sys.exit(1)

    export(args.folder, args.output)


if __name__ == "__main__":
    main()
