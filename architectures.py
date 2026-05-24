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
import re

def _clean_response(text: str) -> str:
    """Usuwa artefakty dialogowe z odpowiedzi modelu."""
    # Usuń linie zaczynające się od "Osoba N:" lub "Agent X:"
    lines = text.split('\n')
    clean_lines = []
    skip_next_empty = False
    for line in lines:
        if re.match(r'^(Osoba\s*\d+|[A-ZŻŹĆĄŚĘŁÓŃ][a-zżźćąśęłóń]+)\s*:', line):
            skip_next_empty = True
            continue
        if skip_next_empty and line.strip() == '':
            skip_next_empty = False
            continue
        skip_next_empty = False
        clean_lines.append(line)
    result = '\n'.join(clean_lines).strip()
    # Jeśli po czyszczeniu zostało cokolwiek sensownego, zwróć to
    # Jeśli nie — zwróć oryginalny tekst żeby nie stracić danych
    return result if len(result) > 20 else text

def round_robin(agents, topic, num_rounds, config):
    """Każdy agent odpowiada po kolei. Wszyscy widzą całą historię."""
    debate_log = []
    history = [f"Temat debaty: {topic}"]

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Runda {round_num} ---")
        for agent in agents:
            history_len = len(history)
            text, tokens = agent.respond(history, config)
            clean_text = _clean_response(text)
            debate_log.append({
                "agent": agent.name,
                "round": round_num,
                "text": clean_text,      # ← czysty tekst do metryk
                "text_raw": text,
                "tokens": tokens,
                "history_len": history_len,
            })
            history.append(f"{agent.name}: {text}")
            print(f"[{agent.name}] ({tokens} tok): {text[:120]}{'...' if len(text) > 120 else ''}")

    return debate_log

ARCHITECTURES = {
    "round_robin": round_robin
}