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
"""

def round_robin(agents, topic, num_rounds, config):
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
            print(f"\n[{agent.name}] ({tokens} tok):\n{text}")

    return debate_log

ARCHITECTURES = {
    "round_robin": round_robin
}