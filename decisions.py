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
        # dla judge:
        "judge_tokens": int,
    }
"""

from collections import Counter
from agents import _generate


# =============================================================================
# 1. JUDGE
# =============================================================================
def judge_decision(agents, judge, debate_log, topic, config):
    print("\n--- Protokół: judge ---")
    verdict, tokens = judge.summarize(debate_log, topic, config)
    print(f"[Sędzia] ({tokens} tok): {verdict[:120]}{'...' if len(verdict) > 120 else ''}")
    return {
        "protocol": "judge",
        "final_answer": verdict,
        "judge_tokens": tokens,
    }


# =============================================================================
# 2. VOTING
# =============================================================================
def voting_decision(agents, judge, debate_log, topic, config):
    print("\n--- Protokół: voting ---")
    transcript = _format_transcript(debate_log, topic)

    # Krok 1: finalne propozycje
    proposals = {}
    proposal_tokens = {}
    for agent in agents:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{transcript}\n\n"
                    "Na podstawie powyższej debaty sformułuj swoją FINALNĄ odpowiedź "
                    "w jednym zdaniu. Odpowiedz tylko tym zdaniem, bez wyjaśnień."
                ),
            },
        ]
        text, tokens = _generate(agent.model, agent.tokenizer, messages, config)
        proposals[agent.name] = text
        proposal_tokens[agent.name] = tokens
        print(f"  Propozycja [{agent.name}]: {text[:100]}...")

    # Krok 2: głosowanie
    options = list(proposals.values())
    options_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(options))
    votes = {}

    for agent in agents:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Temat: {topic}\n\nPropozycje finalne:\n{options_str}\n\n"
                    f"Zagłosuj na NAJLEPSZĄ odpowiedź. Odpowiedz wyłącznie liczbą (1-{len(options)})."
                ),
            },
        ]
        vote_text, _ = _generate(agent.model, agent.tokenizer, messages, config)
        vote_idx = _parse_vote(vote_text, num_options=len(options))
        votes[agent.name] = vote_idx
        print(f"  Głos [{agent.name}]: opcja {vote_idx + 1}")

    # Krok 3: wynik
    tally = Counter(options[idx] for idx in votes.values())
    winner = tally.most_common(1)[0][0]
    print(f"  Zwycięzca: {winner[:100]}...")

    return {
        "protocol": "voting",
        "final_answer": winner,
        "proposals": proposals,
        "proposal_tokens": proposal_tokens,
        "votes": votes,                        # {agent: indeks opcji 0-based}
        "tally": dict(tally),
    }


# =============================================================================
# 3. CONSENSUS
# =============================================================================
def consensus_decision(agents, judge, debate_log, topic, config):
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
                        "Sformułuj propozycję wspólnego stanowiska wszystkich agentów "
                        "w jednym zdaniu, które mogłoby ich pogodzić."
                    ),
                },
            ]
            current_proposal, _ = _generate(agents[0].model, agents[0].tokenizer, messages, config)

        round_data["proposal"] = current_proposal
        print(f"  Runda {round_num}: {current_proposal[:100]}...")

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
    "judge": judge_decision,
    "voting": voting_decision,
    "consensus": consensus_decision,
}