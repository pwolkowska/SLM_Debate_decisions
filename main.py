"""
main.py — Punkt wejścia.

Uruchomienie:
    python main.py                              # używa config.yaml
    python main.py --output wyniki/moj_test     # własna nazwa pliku wyjściowego (bez rozszerzenia)

Co robi:
  1. Wczytuje config.yaml
  2. Ładuje model z Hugging Face
  3. Tworzy agentów i sędziego
  4. Uruchamia wybraną architekturę debaty
  5. Uruchamia wybrany protokół decyzyjny
  6. Liczy metryki
  7. Zapisuje wyniki do dwóch plików:
       <output>.json  — kompletne dane ustrukturyzowane (debata + metryki)
       <output>.txt   — czytelny podgląd dla człowieka
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents import Agent, Judge
from architectures import ARCHITECTURES
from decisions import DECISIONS
from metrics import compute_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ścieżka wyjściowa bez rozszerzenia (np. wyniki/test_1). "
             "Domyślnie: wyniki/<timestamp>"
    )
    args = parser.parse_args()

    # 1. Config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ustal ścieżkę wyjściową
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("wyniki") / ts
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Model
    device = config.get("device", "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Ładowanie modelu: {config['model_name']} na {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=dtype).to(device)
    print("Model załadowany.\n")

    # 3. Agenci
    agents = [
        Agent(name=a["name"], system_prompt=a["system_prompt"], model=model, tokenizer=tokenizer)
        for a in config["agents"]
    ]
    judge = Judge(
        system_prompt=config["judge"]["system_prompt"], model=model, tokenizer=tokenizer
    )

    # 4. Debata
    arch_name = config["architecture"]
    decision_name = config.get("decision_protocol", "judge")
    print(f"Architektura: {arch_name} | Protokół: {decision_name}")
    print(f"Temat: {config['topic']}")
    print(f"Agenci: {', '.join(a.name for a in agents)}\n")

    debate_log = ARCHITECTURES[arch_name](agents, judge, config["topic"], config["num_rounds"], config)

    # 5. Decyzja
    decision_result = DECISIONS[decision_name](agents, judge, debate_log, config["topic"], config)

    # 6. Metryki
    print("\nLiczę metryki...")
    metryki = compute_all(debate_log, decision_result, config)

    # 7. Zapis
    result = _build_result(config, debate_log, decision_result, metryki, args.output)
    _save_json(result, out_path)
    _save_txt(result, out_path)

    print(f"\nZapisano:")
    print(f"  {out_path}.json")
    print(f"  {out_path}.txt")
    print(f"\nOstateczna decyzja: {decision_result['final_answer'][:200]}")


def _build_result(config, debate_log, decision_result, metryki, output_arg):
    """Buduje kompletny słownik wyników."""
    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_name": output_arg,
            "model": config["model_name"],
            "device": config.get("device", "cpu"),
            "architecture": config["architecture"],
            "protocol": config.get("decision_protocol", "judge"),
            "num_rounds": config["num_rounds"],
            "topic": config["topic"],
            "agents": [
                {"name": a["name"], "system_prompt": a["system_prompt"]}
                for a in config["agents"]
            ],
            "generation": {
                "temperature": config.get("temperature"),
                "max_new_tokens": config.get("max_new_tokens"),
                "do_sample": config.get("do_sample"),
            },
            "consensus_threshold": config.get("consensus_threshold"),
            "max_consensus_rounds": config.get("max_consensus_rounds"),
        },
        "debate": debate_log,
        "decision": decision_result,
        "metrics": metryki,
    }


def _save_json(result: dict, out_path: Path):
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _save_txt(result: dict, out_path: Path):
    """Czytelny podgląd dla człowieka."""
    txt_path = out_path.with_suffix(".txt")
    meta = result["meta"]
    lines = []

    lines.append("=" * 70)
    lines.append(f"  DEBATA — {meta['timestamp']}")
    lines.append("=" * 70)
    lines.append(f"Temat:        {meta['topic']}")
    lines.append(f"Agenci:       {', '.join(a['name'] for a in meta['agents'])}")
    lines.append(f"Architektura: {meta['architecture']}  |  Protokół: {meta['protocol']}")
    lines.append(f"Rundy:        {meta['num_rounds']}  |  Model: {meta['model']}")
    lines.append("")

    # Wypowiedzi
    current_round = None
    for e in result["debate"]:
        if e["round"] != current_round:
            current_round = e["round"]
            lines.append(f"\n{'─' * 70}")
            lines.append(f"  RUNDA {current_round}")
            lines.append(f"{'─' * 70}")
        lines.append(f"\n[{e['agent']}] ({e['tokens']} tokenów):")
        lines.append(e["text"])

    # Decyzja
    dec = result["decision"]
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  PROTOKÓŁ: {dec['protocol'].upper()}")
    lines.append(f"{'=' * 70}")

    if dec["protocol"] == "consensus":
        for r in dec.get("consensus_rounds", []):
            lines.append(f"\nRunda konsensusu {r['round']}:")
            lines.append(f"  Propozycja: {r['proposal']}")
            for agent, zgoda in r["votes"].items():
                lines.append(f"  [{agent}]: {'TAK' if zgoda else 'NIE'}")
            lines.append(f"  Zgoda: {r['agreement_ratio']:.0%}  |  Osiągnięty: {r['reached']}")

    elif dec["protocol"] == "voting":
        lines.append("\nPropozycje:")
        for agent, prop in dec.get("proposals", {}).items():
            lines.append(f"  [{agent}]: {prop}")
        lines.append("\nGłosy:")
        for agent, idx in dec.get("votes", {}).items():
            lines.append(f"  [{agent}]: opcja {idx + 1}")

    lines.append(f"\nOSTATECZNA DECYZJA:\n{dec['final_answer']}")

    # Metryki — skrócony widok
    m = result["metrics"]
    lines.append(f"\n{'=' * 70}")
    lines.append("  METRYKI")
    lines.append(f"{'=' * 70}")

    tpt = m["tokens_per_turn"]["overall"]
    lines.append(f"\nTokeny per wypowiedź:  średnia={tpt['mean']}  std={tpt['std']}  łącznie={tpt['total']}")

    lines.append("\nTokeny per agent:")
    for agent, stats in m["tokens_per_turn"]["per_agent"].items():
        lines.append(f"  {agent}: średnia={stats['mean']}  std={stats['std']}")

    lines.append("\nFlip Rate (zmiana stanowiska):")
    for agent, fr in m["flip_rate"].items():
        tof = fr["ToF"] if fr["ToF"] else "—"
        lines.append(f"  {agent}: NoF={fr['NoF']}  ToF={tof}  rundy={fr['flip_rounds']}")

    lines.append(f"\nEntropia per runda: {m['entropy']['per_round']}  (średnia: {m['entropy']['mean']})")

    conv = m["convergence"]
    lines.append(f"\nConvergence Round: {conv['convergence_round']}  (metoda: {conv['method']}  osiągnięty: {conv['reached']})")

    auc = m["auc_agreement"]
    lines.append(f"\nAUC-Agreement: {auc['auc']}  per runda: {auc['per_round']}")

    sd = m["semantic_diversity"]
    if sd.get("overall") is not None:
        lines.append(f"\nSemantic Diversity: {sd['overall']}  per runda: {sd['per_round']}")
    else:
        lines.append(f"\nSemantic Diversity: niedostępne ({sd.get('note', '')})")

    rr = m["redundancy_ratio"]
    if rr.get("mean") is not None:
        lines.append(f"\nRedundancy Ratio: {rr['mean']}  per runda: {rr['per_round']}")
    else:
        lines.append(f"\nRedundancy Ratio: niedostępne ({rr.get('note', '')})")

    lines.append("")
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()