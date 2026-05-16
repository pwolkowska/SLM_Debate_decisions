"""
architectures.py — Trzy architektury wymiany informacji w debacie.

Każda funkcja zwraca debate_log — listę dict z polami:
    {
        "agent": str,
        "round": int,
        "text": str,
        "tokens": int,          # liczba wygenerowanych tokenów
        "history_len": int,     # ile wypowiedzi agent widział przed odpowiedzią
    }

Architektury:
    round_robin  — wszyscy widzą całą historię, stała kolejność
    relay        — każdy widzi TYLKO wypowiedź poprzedniego (głuchy telefon)
    free_for_all — wszyscy widzą całą historię, losowa kolejność w każdej rundzie
"""

import random


def round_robin(agents, judge, topic, num_rounds, config):
    """Każdy agent odpowiada po kolei. Wszyscy widzą całą historię."""
    debate_log = []
    history = [f"Temat debaty: {topic}"]

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Runda {round_num} ---")
        for agent in agents:
            history_len = len(history)
            text, tokens = agent.respond(history, config)
            debate_log.append({
                "agent": agent.name,
                "round": round_num,
                "text": text,
                "tokens": tokens,
                "history_len": history_len,
            })
            history.append(f"{agent.name}: {text}")
            print(f"[{agent.name}] ({tokens} tok): {text[:120]}{'...' if len(text) > 120 else ''}")

    return debate_log


def relay(agents, judge, topic, num_rounds, config):
    """Łańcuch — każdy agent widzi TYLKO wypowiedź poprzedniego agenta + temat."""
    debate_log = []
    previous_response = f"Temat debaty: {topic}"

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Runda {round_num} ---")
        for agent in agents:
            history = [f"Temat debaty: {topic}", previous_response]
            text, tokens = agent.respond(history, config)
            debate_log.append({
                "agent": agent.name,
                "round": round_num,
                "text": text,
                "tokens": tokens,
                "history_len": 2,
            })
            previous_response = f"{agent.name}: {text}"
            print(f"[{agent.name}] ({tokens} tok): {text[:120]}{'...' if len(text) > 120 else ''}")

    return debate_log


def free_for_all(agents, judge, topic, num_rounds, config):
    """Jak round_robin, ale kolejność agentów jest losowa w każdej rundzie."""
    debate_log = []
    history = [f"Temat debaty: {topic}"]

    for round_num in range(1, num_rounds + 1):
        shuffled = list(agents)
        random.shuffle(shuffled)
        print(f"\n--- Runda {round_num} (kolejność: {', '.join(a.name for a in shuffled)}) ---")

        for agent in shuffled:
            history_len = len(history)
            text, tokens = agent.respond(history, config)
            debate_log.append({
                "agent": agent.name,
                "round": round_num,
                "text": text,
                "tokens": tokens,
                "history_len": history_len,
            })
            history.append(f"{agent.name}: {text}")
            print(f"[{agent.name}] ({tokens} tok): {text[:120]}{'...' if len(text) > 120 else ''}")

    return debate_log


ARCHITECTURES = {
    "round_robin": round_robin,
    "relay": relay,
    "free_for_all": free_for_all,
}