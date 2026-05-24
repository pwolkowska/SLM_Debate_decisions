"""
decisions.py — Protokoły wyboru ostatecznej decyzji po debacie.

Każda funkcja zwraca słownik:
    {
        "final_answer": str,
        "protocol": str,
        # dla voting:
        "proposals": {agent_name: str},
        "votes": {agent_name: int},          # indeks wybranej opcji (0-based)
        "tally": {propozycja: liczba_głosów},
        # dla consensus:
        "consensus_rounds": [
            {
                "round": int,
                "proposal": str,
                "votes": {agent_name: bool},
                "agreement_ratio": float,
                "reached": bool,
            }
        ],
    }
"""

from collections import Counter
from agents import _generate

def consensus_decision(agents, debate_log, topic, config):
    threshold = config.get("consensus_threshold", 0.66)
    max_rounds = config.get("max_consensus_rounds", 3)
    print(f"\n--- Protokół: consensus (próg: {threshold:.0%}) ---")

    # Skrócona historia — tylko ostatnie wypowiedzi, bez przepisywania
    transcript = _format_transcript_short(debate_log, topic)
    current_proposal = None
    consensus_rounds = []

    for round_num in range(1, max_rounds + 1):
        round_data = {"round": round_num, "proposal": None, "votes": {}, "agreement_ratio": 0.0, "reached": False}

        # Krok A: propozycja
        if current_proposal is None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Jesteś mediatorem. Formułujesz krótkie, neutralne propozycje kompromisu. "
                        "Piszesz TYLKO jedno zdanie. Bez wstępu, bez podsumowania, bez formatowania, "
                        "bez gwiazdek, bez cudzysłowów, bez imion rozmówców."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Główne argumenty stron:\n{transcript}\n\n"
                        "Napisz JEDNO zdanie kompromisu. Samo zdanie, nic więcej."
                    ),
                },
            ]
            current_proposal, _ = _generate(agents[0].model, agents[0].tokenizer, messages, config)
            current_proposal = _clean_proposal(current_proposal)

        round_data["proposal"] = current_proposal
        print(f"  Runda {round_num}: {current_proposal[:100]}...")

        # Krok B: głosowanie TAK/NIE
        agreements = []
        for agent in agents:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{agent.system_prompt} "
                        "Odpowiadasz jednym słowem: TAK albo NIE. Nic więcej."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Propozycja kompromisu: {current_proposal}\n\n"
                        "Czy akceptujesz tę propozycję? Odpowiedz jednym słowem: TAK albo NIE."
                    ),
                },
            ]
            response, _ = _generate(agent.model, agent.tokenizer, messages, config)
            agrees = _parse_yes_no(response)
            agreements.append(agrees)
            round_data["votes"][agent.name] = agrees
            print(f"    [{agent.name}]: {'TAK' if agrees else 'NIE'}")

        ratio = sum(agreements) / len(agreements)
        round_data["agreement_ratio"] = ratio

        if ratio >= threshold:
            round_data["reached"] = True
            consensus_rounds.append(round_data)
            print(f"  Konsensus osiągnięty ({ratio:.0%})")
            break

        # Krok C: modyfikacja
        dissenters = [a for a, ok in zip(agents, agreements) if not ok]
        if dissenters:
            modifier = dissenters[0]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Jesteś mediatorem. Modyfikujesz propozycję kompromisu. "
                        "Piszesz TYLKO jedno zdanie. Bez wstępu, bez formatowania, bez gwiazdek."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Aktualna propozycja: {current_proposal}\n\n"
                        f"Agent {modifier.name} jej nie akceptuje. "
                        "Napisz JEDNO zmodyfikowane zdanie, które będzie bardziej neutralne. "
                        "Samo zdanie, nic więcej."
                    ),
                },
            ]
            current_proposal, _ = _generate(modifier.model, modifier.tokenizer, messages, config)
            current_proposal = _clean_proposal(current_proposal)

        consensus_rounds.append(round_data)

    else:
        print(f"  Brak konsensusu po {max_rounds} rundach — zwracamy ostatnią propozycję")

    return {
        "protocol": "consensus",
        "final_answer": current_proposal,
        "consensus_rounds": consensus_rounds,
        "consensus_reached": any(r["reached"] for r in consensus_rounds),
        "convergence_round": next((r["round"] for r in consensus_rounds if r["reached"]), None),
    }


# =============================================================================
# Helpery
# =============================================================================

def _clean_proposal(text: str) -> str:
    """Usuwa markdown i bierze tylko pierwsze zdanie."""
    import re
    # usuń markdown
    text = re.sub(r'\*\*?(.+?)\*\*?', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = text.strip().strip('"').strip("'")
    # weź tylko pierwsze zdanie
    match = re.search(r'^[^.!?]+[.!?]', text)
    return match.group(0).strip() if match else text.split('\n')[0].strip()

def _parse_yes_no(text):
    t = text.strip().lower()
    if t.startswith("tak") or t.startswith("yes"):
        return True
    if t.startswith("nie") or t.startswith("no"):
        return False
    head = t[:30]
    if "tak" in head or "yes" in head:
        return True
    return False


def _format_transcript(debate_log, topic):
    out = f"Temat debaty: {topic}\n\n"
    for entry in debate_log:
        out += f"[Runda {entry['round']}] {entry['agent']}: {entry['text']}\n\n"
    return out

def _format_transcript_short(debate_log: list, topic: str) -> str:
    """Pełna historia debaty, ale oczyszczona z markdown."""
    import re
    lines = []
    for entry in debate_log:
        text = re.sub(r'\*\*?(.+?)\*\*?', r'\1', entry["text"])  # usuń bold
        text = re.sub(r'#+\s*', '', text)                          # usuń nagłówki
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # usuń listy
        text = ' '.join(text.split())                              # spłaszcz whitespace
        lines.append(f"[Runda {entry['round']}] {entry['agent']}: {text}")
    return '\n\n'.join(lines)

DECISIONS = {
    "consensus": consensus_decision,
}