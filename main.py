"""
main.py — Punkt wejścia.

Uruchomienie:
    python main.py                              # używa config.yaml
    python main.py --output wyniki/moj_test     # własna nazwa pliku wyjściowego (bez rozszerzenia)

Co robi:
  1. Wczytuje config.yaml
  2. Ładuje model z Hugging Face
  3. Tworzy agentów
  4. Uruchamia wybraną architekturę debaty
  5. Uruchamia wybrany protokół decyzyjny
  6. Liczy metryki
  7. Zapisuje wyniki do dwóch plików:
       <output>.json  — kompletne dane ustrukturyzowane (debata + metryki)
       <output>.txt   — czytelny podgląd dla człowieka
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents import Agent
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

    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

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

    # 4. Debata
    arch_name = config["architecture"]
    decision_name = config.get("decision_protocol", "consensus")
    print(f"Architektura: {arch_name} | Protokół: {decision_name}")
    print(f"Temat: {config['topic']}")
    print(f"Agenci: {', '.join(a.name for a in agents)}\n")

    debate_log = ARCHITECTURES[arch_name](agents, config["topic"], config["num_rounds"], config)

    # 5. Decyzja
    decision_result = DECISIONS[decision_name](agents, debate_log, config["topic"], config)

    # 6. Metryki
    print("\nLiczę metryki...")
    metryki, metryki_j = compute_all(debate_log, decision_result, config)

    # 7. Zapis
    result = _build_result(config, debate_log, decision_result, metryki, metryki_j, args.output)
    _save_json(result, out_path)
    _save_txt(result, out_path)

    print(f"\nZapisano:")
    print(f"  {out_path}.json")
    print(f"  {out_path}.txt")
    print(f"\nOstateczna decyzja: {decision_result['final_answer'][:200]}")


def _build_result(config, debate_log, decision_result, metryki, metryki_j, output_arg):
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
                "seed": config.get("seed"),
            },
            "consensus_threshold": config.get("consensus_threshold"),
            "max_consensus_rounds": config.get("max_consensus_rounds"),
        },
        "debate": debate_log,
        "decision": decision_result,
        "metrics": metryki,
        "metrics_j": metryki_j,
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

    # tokens_per_turn
    tp = m["tokens_per_turn"]
    ov = tp["overall"]
    lines.append(f"\nTokeny per wypowiedź: średnia={ov['mean']}  mediana={ov['median']}  łącznie={ov['total']}")
    lines.append("  Per agent:")
    for agent, stats in tp["per_agent"].items():
        lines.append(f"    {agent}: średnia={stats['mean']}  total={stats['total']}")

    # convergence
    conv = m["convergence"]
    lines.append(f"\nKonsensus: osiągnięty={conv['reached']}  runda={conv['convergence_round']}")
    lines.append(f"  Frakcje zgody: {conv['agreement_ratios']}")

    # opinion_shift
    shift = m["opinion_shift"]
    if shift.get("mean") is not None:
        lines.append(f"\nZmiana stanowiska (opinion shift):")
        for agent, val in shift["per_agent"].items():
            lines.append(f"  {agent}: {val}")
        lines.append(f"  Średnia: {shift['mean']}")
    else:
        lines.append(f"\nZmiana stanowiska: niedostępne ({shift.get('note', '')})")

    # between_agent_similarity
    bas = m["between_agent_similarity"]
    if bas.get("initial") is not None:
        lines.append(f"\nPodobieństwo między agentami:")
        lines.append(f"  Runda 1 (initial): {bas['initial']}  Ostatnia (final): {bas['final']}  Delta: {bas['delta']}")
        lines.append(f"  Per runda: {bas['per_round']}")
    else:
        lines.append(f"\nPodobieństwo między agentami: niedostępne ({bas.get('note', '')})")

    # lexical_richness
    lr = m["lexical_richness"]
    lines.append(f"\nBogactwo leksykalne (TTR):")
    for agent, stats in lr["per_agent"].items():
        lines.append(f"  {agent}: TTR={stats['ttr']}  unikalne={stats['unique_words']}  łącznie={stats['total_words']}")

    # --- METRYKI_J ---
    mj = result.get("metrics_j", {})
    if mj:
        lines.append(f"\n{'=' * 70}")
        lines.append("  METRYKI_J")
        lines.append(f"{'=' * 70}")

        # argument_novelty
        an = mj.get("argument_novelty", {})
        if an.get("overall") is not None:
            lines.append(f"\nNowość argumentów (argument novelty): overall={an['overall']}")
            for agent, stats in an.get("per_agent", {}).items():
                lines.append(f"  {agent}: średnia={stats['mean']}  std={stats['std']}")
        else:
            lines.append(f"\nNowość argumentów: niedostępne ({an.get('note', '')})")

        # claim_grounding
        cg = mj.get("claim_grounding", {})
        lines.append(f"\nUgruntowanie twierdzeń (claim grounding) [{cg.get('mode_used', '?')}]:"
                     f"  overall={cg.get('overall')}")
        for agent, stats in cg.get("per_agent", {}).items():
            lines.append(f"  {agent}: rate={stats['grounding_rate']}  "
                         f"zdania={stats['grounded_sentences']}/{stats['total_sentences']}")

        # question_density
        qd = mj.get("question_density", {})
        lines.append(f"\nZagęszczenie pytań: overall={qd.get('overall')}")
        for agent, stats in qd.get("per_agent", {}).items():
            lines.append(f"  {agent}: rate={stats['question_rate']}  pytania={stats['question_count']}")

        # direct_address_rate
        da = mj.get("direct_address_rate", {})
        lines.append(f"\nBezpośrednie adresowanie: średnia={da.get('mean_address_rate')}  "
                     f"najczęściej adresowany={da.get('most_addressed')}")
        for agent, stats in da.get("per_agent", {}).items():
            lines.append(f"  {agent}: rate={stats['address_rate']}  "
                         f"adresował={dict(stats.get('addressed_agents', {}))}")

        # rebuttal_depth
        rd = mj.get("rebuttal_depth", {})
        if rd.get("overall") is not None:
            lines.append(f"\nGłębokość riposty (rebuttal depth): overall={rd['overall']}")
            for agent, stats in rd.get("per_agent", {}).items():
                lines.append(f"  {agent}: mean_sim={stats['mean_rebuttal_sim']}")
        else:
            lines.append(f"\nGłębokość riposty: niedostępne ({rd.get('note', '')})")

        # reciprocal_shift_index
        rs = mj.get("reciprocal_shift_index", {})
        if rs.get("mean_abs_r") is not None:
            lines.append(f"\nReaktywność debaty (reciprocal shift): mean |r|={rs['mean_abs_r']}")
            for pair, stats in rs.get("per_pair", {}).items():
                r_val = stats.get("pearson_r")
                note = f"  ({stats['note']})" if "note" in stats else ""
                lines.append(f"  {pair}: r={r_val}  rundy={stats.get('n_rounds')}{note}")
        else:
            lines.append(f"\nReaktywność debaty: niedostępne ({rs.get('note', '')})")

        # opinion_trajectory_monotonicity
        otm = mj.get("opinion_trajectory_monotonicity", {})
        lines.append(f"\nMonotoniczność trajektorii: overall_abs={otm.get('overall_abs')}"
                     + (f"  [{otm['note']}]" if "note" in otm else ""))
        for agent, stats in otm.get("per_agent", {}).items():
            if stats.get("monotonicity") is not None:
                lines.append(f"  {agent}: M={stats['monotonicity']}  "
                             f"zmiany_kierunku={stats.get('direction_changes')}")

        # convergence_speed
        cs = mj.get("convergence_speed", {})
        if cs.get("note"):
            lines.append(f"\nSzybkość konwergencji: niedostępne ({cs['note']})")
        else:
            lines.append(f"\nSzybkość konwergencji (próg={cs.get('threshold')}): "
                         f"osiągnięta={cs.get('reached')}  runda={cs.get('convergence_round')}")
            lines.append(f"  initial_sim={cs.get('initial_similarity')}  "
                         f"final_sim={cs.get('final_similarity')}")

        # hedging_rate
        hr = mj.get("hedging_rate", {})
        lines.append(f"\nHedging [{hr.get('mode_used', '?')}]: overall={hr.get('overall')}")
        for agent, stats in hr.get("per_agent", {}).items():
            lines.append(f"  {agent}: rate={stats['hedging_rate']}  "
                         f"zdania={stats['hedged_sentences']}/{stats['total_sentences']}")

        # assertiveness_score
        asc = mj.get("assertiveness_score", {})
        lines.append(f"\nAsertywność [{asc.get('mode_used', '?')}]: overall={asc.get('overall')}")
        for agent, stats in asc.get("per_agent", {}).items():
            lines.append(f"  {agent}: assertiveness={stats['assertiveness_rate']}  "
                         f"hedging={stats['hedging_rate']}  net={stats['net_assertiveness']}")

        # turn_length_entropy
        tle = mj.get("turn_length_entropy", {})
        lines.append(f"\nEntropia długości tur: overall={tle.get('overall')} bitów")
        for agent, stats in tle.get("per_agent", {}).items():
            if stats.get("entropy_bits") is not None:
                lines.append(f"  {agent}: H={stats['entropy_bits']} b  "
                             f"min={stats['min']}  max={stats['max']}  mean={stats['mean']}")

        # gini_speaking_time
        gs = mj.get("gini_speaking_time", {})
        lines.append(f"\nNierówność czasu mówienia (Gini): {gs.get('gini')}  "
                     f"dominuje={gs.get('dominant_agent')}")
        for agent, share in gs.get("per_agent_share", {}).items():
            tokens = gs.get("per_agent_tokens", {}).get(agent, "?")
            lines.append(f"  {agent}: {share:.1%}  ({tokens} tokenów)")

    lines.append("")
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()