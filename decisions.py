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

# =============================================================================
# CONSENSUS
# =============================================================================
def consensus_decision(agents, debate_log, topic, config):
    threshold = config.get("consensus_threshold", 0.66)
    max_rounds = config.get("max_consensus_rounds", 3)
    print(f"\n--- Protokół: consensus (próg: {threshold:.0%}) ---")

    transcript = _format_transcript(debate_log, topic)
    current_proposal = None
    consensus_rounds = []

    for round_num in range(1, max_rounds + 1):
        round_data = {"round": round_num, "proposal": None, "votes": {}, "agreement_ratio": 0.0, "reached": False}

        # Krok A: propozycja
        if current_proposal is None:
            messages = [
                {"role": "system", "content": agents[0].system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{transcript}\n\n"
                        #"Sformułuj propozycję wspólnego stanowiska wszystkich agentów "
                        #"w jednym zdaniu, które mogłoby ich pogodzić."
                        "Zaproponuj wspólne stanowisko, "
                        "które byłoby akceptowalne dla większości agentów. "
                        "Odpowiedz jednym krótkim zdaniem."
                    ),
                },
            ]
            current_proposal, _ = _generate(agents[0].model, agents[0].tokenizer, messages, config)

        round_data["proposal"] = current_proposal
        print(f"  Runda {round_num}: {current_proposal}")

        # Krok B: głosowanie TAK/NIE
        agreements = []
        for agent in agents:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Proponowane wspólne stanowisko:\n\"{current_proposal}\"\n\n"
                        "Czy ZGADZASZ się z tym stanowiskiem? Odpowiedz tylko TAK lub NIE."
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
                {"role": "system", "content": modifier.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\nAktualne stanowisko:\n\"{current_proposal}\"\n\n"
                        "Zmodyfikuj je tak, aby było bardziej akceptowalne dla wszystkich. "
                        "Odpowiedz jednym zdaniem."
                    ),
                },
            ]
            current_proposal, _ = _generate(modifier.model, modifier.tokenizer, messages, config)

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
def _parse_vote(text, num_options):
    for char in text.strip():
        if char.isdigit():
            num = int(char)
            if 1 <= num <= num_options:
                return num - 1
    return 0


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


DECISIONS = {
    "consensus": consensus_decision,
}