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
                "seed": config.get("seed"),
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

    # opinion_shift_sentiment
    oss = m.get("opinion_shift_sentiment", {})
    lines.append(f"\nZmiana sentymentu (opinion_shift_sentiment):")
    if oss.get("note"):
        lines.append(f"  niedostępne ({oss['note']})")
    else:
        for agent, stats in oss.get("per_agent", {}).items():
            if "error" in stats:
                lines.append(f"  {agent}: błąd — {stats['error']}")
                continue
            traj = "  →  ".join(
                f"R{t['round']}:{t['sentiment']:+.4f}" for t in stats["trajectory"]
            )
            lines.append(
                f"  {agent}: Δ={stats['delta_first_last']}  "
                f"mean|shift|={stats['mean_shift_abs']}  [{traj}]"
            )

    # between_agent_similarity_sentiment
    bass = m.get("between_agent_similarity_sentiment", {})
    lines.append(f"\nPodobieństwo sentymentu między agentami:")
    if bass.get("note"):
        lines.append(f"  niedostępne ({bass['note']})")
    else:
        lines.append(
            f"  Runda 1 (initial): {bass.get('initial')}  "
            f"Ostatnia (final): {bass.get('final')}  Delta: {bass.get('delta')}"
        )
        for rnd, rdata in bass.get("per_round", {}).items():
            if "error" in rdata:
                lines.append(f"  Runda {rnd}: błąd — {rdata['error']}")
                continue
            ag_sent = "  ".join(
                f"{ag}:{v['sentiment']:+.4f}{'[rep]' if v['is_repetition'] else ''}"
                for ag, v in rdata["agent_sentiments"].items()
            )
            reps = rdata.get("agents_with_repetition", [])
            rep_note = f"  (powtórzenia: {', '.join(reps)})" if reps else ""
            lines.append(
                f"  Runda {rnd}: mean_pair_diff={rdata['mean_pair_diff']}  "
                f"[{ag_sent}]{rep_note}"
            )

    # answer_flip_rate
    afr = m.get("answer_flip_rate", {})
    lines.append(f"\nAnswer Flip Rate (NoF / ToF):")
    if afr.get("note"):
        lines.append(f"  niedostępne ({afr['note']})")
    else:
        summ = afr.get("summary", {})
        lines.append(
            f"  Próg flip: {summ.get('flip_threshold')}  "
            f"mean_NoF={summ.get('mean_nof')}  max_NoF={summ.get('max_nof')}  "
            f"agentów z flipem: {summ.get('agents_with_flip')}"
        )
        for agent, stats in afr.get("per_agent", {}).items():
            if stats.get("note"):
                lines.append(f"  {agent}: {stats['note']}")
                continue
            traj = "  ".join(
                f"R{t['round']}:{'⚑' if t['flip'] else '·'}"
                f"(d={t['cos_dist_prev']})" if t["cos_dist_prev"] is not None
                else f"R{t['round']}:start"
                for t in stats["trajectory"]
            )
            lines.append(
                f"  {agent}: NoF={stats['nof']}  ToF={stats['tof']}  "
                f"flipy w rundach: {stats['flip_rounds']}"
            )
            lines.append(f"    trajektoria: {traj}")

    # history_repetition
    hre = m.get("history_repetition", {})
    lines.append(f"\nPowtarzanie historii (history_repetition):")
    summ = hre.get("summary", {})
    lines.append(
        f"  Łącznie: {summ.get('total_repetitions')}/{summ.get('total_turns')}  "
        f"rate={summ.get('overall_rate')}  "
        f"(próg n-gram={summ.get('ngram_threshold')}, "
        f"próg embed={summ.get('embed_threshold')}, "
        f"prefix={summ.get('prefix_words')} słów)"
    )
    lines.append("  Per agent:")
    for agent, stats in hre.get("per_agent", {}).items():
        lines.append(
            f"    {agent}: powtórzeń={stats['repetition_count']}/{stats['total_turns']}  "
            f"rate={stats['repetition_rate']}"
        )
    lines.append("  Per wypowiedź (wykryte powtórzenia):")
    for e in hre.get("per_entry", []):
        if e["ngram_hit"] or e["embed_hit"]:
            lines.append(
                f"    Runda {e['round']} [{e['agent']}]: "
                f"ngram={e['max_ngram']}  cosine={e['max_cosine']}"
            )
    # semantic_diversity
    sd = m.get("semantic_diversity", {})
    if sd.get("overall") is not None:
        lines.append(f"\nSemantyczna różnorodność debaty (semantic_diversity):")
        lines.append(f"  Overall: {sd['overall']}  (0=jednorodna, 1=różnorodna)")
        lines.append(f"  Per runda: {sd.get('per_round')}")
    else:
        lines.append(f"\nSemantyczna różnorodność: niedostępne ({sd.get('note', '')})")

    # redundancy_ratio
    rr = m.get("redundancy_ratio", {})
    if rr.get("mean") is not None:
        lines.append(f"\nRedundancja (redundancy_ratio, próg={rr.get('threshold')}):")
        lines.append(f"  Średnia: {rr['mean']}  (0=nowe argumenty, 1=powtarzanie)")
        lines.append(f"  Per runda: {rr.get('per_round')}")
    else:
        lines.append(f"\nRedundancja: niedostępne ({rr.get('note', '')})")


    lines.append("")
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()