"""
automatyzacja.py — Uruchamia eksperymenty dla wszystkich par agentów.

Dla każdej pary agentów i każdej próby:
  1. Nadpisuje config.yaml
  2. Wywołuje main.py z --output wskazującym na docelowy plik
  3. main.py sam zapisuje <nazwa>.json i <nazwa>.txt

Wyniki lądują w folderze wyniki/:
    wyniki/
        pesymista-optymista_1.json
        pesymista-optymista_1.txt
        pesymista-optymista_2.json
        ...
"""

import os
import subprocess
import yaml
from pathlib import Path

# =====================================================================
# SUFFIXSY PROMPTÓW
# =====================================================================
SUFFIX_AGENT_1 = (
    " Zanim wyrazisz swoje zdanie, zapoznaj się z wypowiedzią drugiego rozmówcy. "
    "Jeśli historia jest pusta, to znaczy, że zaczynasz dyskusję — wówczas przedstaw "
    "swój punkt widzenia w sposób możliwie przekonujący dla Twojego rozmówcy. "
    "Waszym celem jest możliwie szybko wspólnie znaleźć najlepsze rozwiązanie zadanego tematu."
)

SUFFIX_AGENT_2 = (
    " Prowadzisz rozmowę i szukasz wspólnej odpowiedzi. "
    "Zanim wyrazisz swoje zdanie, zapoznaj się z wypowiedzią drugiego rozmówcy. "
    "Waszym celem jest możliwie szybko znaleźć najlepsze rozwiązanie zadanego tematu."
)

# =====================================================================
# PARY AGENTÓW
# =====================================================================
pary_agentow = {
    "ryzykant-asekurant": [
        {"name": "Ryzykant", "cecha": "Jesteś osobą bardzo odważną i czerpiesz satysfakcję z podejmowania niebezpiecznych, ekstremalnych decyzji."},
        {"name": "Asekurant", "cecha": "Jesteś osobą lękliwą, skrajnie ostrożną i nieufną wobec nowości."},
    ],
    # "pesymista-optymista": [
    #     {"name": "Pesymista", "cecha": "Jesteś osobą ostrożną i sceptyczną. Koncentrujesz się na tym, co może pójść nie tak. Masz skłonność do analizowania najgorszych możliwych scenariuszy i traktujesz je jako punkt wyjścia do oceny sytuacji."},
    #     {"name": "Optymista", "cecha": "Jesteś pozytywnie nastawiony do świata i ludzi. Wierzysz, że nawet trudne sytuacje mają w sobie potencjał do poprawy. Unikasz czarnych scenariuszy."},
    # ],
    # "emocjonalny-racjonalny": [
    #     {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
    #     {"name": "Racjonalny",  "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    # ],
    # "racjonalny-emocjonalny": [
    #     {"name": "Racjonalny",  "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    #     {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
    # ],
    # "konfrontacyjny-unikajacy": [
    #     {"name": "Konfrontacyjny",      "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie. Mówisz stanowczo i nie boisz się napięcia w rozmowie."},
    #     {"name": "Unikający konfliktu", "cecha": "Jesteś nastawiony na unikanie konfliktów. Starasz się łagodzić napięcia i utrzymywać dobrą atmosferę w rozmowie."},
    # ],
}

# =====================================================================
# STAŁE USTAWIENIA
# =====================================================================
TOPIC = (
    "Firma ma kłopoty finansowe. Czy lepiej jest zwolnić 30% pracowników, "
    "żeby uratować pozostałych 70%, czy wszystkim obniżyć wypłatę o 20%, ale nikogo nie zwalniać?"
)
ILOSC_POWTORZEN = 2
FOLDER_WYNIKOW = Path("wyniki")


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def buduj_config(agent1: dict, agent2: dict) -> dict:
    return {
        "model_name": "speakleash/Bielik-1.5B-v3.0-Instruct",
        "device": "cpu",
        "temperature": 0.7,
        "max_new_tokens": 256,
        "do_sample": True,
        "architecture": "round_robin",
        "num_rounds": 3,
        "topic": TOPIC,
        "decision_protocol": "consensus",
        "consensus_threshold": 1.0,
        "max_consensus_rounds": 3,
        "agents": [
            {"name": agent1["name"], "system_prompt": agent1["cecha"] + SUFFIX_AGENT_1},
            {"name": agent2["name"], "system_prompt": agent2["cecha"] + SUFFIX_AGENT_2},
        ],
        "judge": {
            "system_prompt": "Jesteś sędzią. Podsumuj debatę i wskaż najlepsze rozwiązanie."
        },
    }


def zapisz_config(config: dict):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def uruchom_probe(output_path: Path, env: dict) -> bool:
    """Uruchamia main.py z --output. Zwraca True jeśli sukces."""
    result = subprocess.run(
        ["python", "-X", "utf8", "main.py", "--output", str(output_path)],
        text=True,
        encoding="utf-8",
        env=env,
    )
    return result.returncode == 0


# =====================================================================
# GŁÓWNA PĘTLA
# =====================================================================

def main():
    FOLDER_WYNIKOW.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    print(f"Rozpoczynam eksperymenty. Wyniki → {FOLDER_WYNIKOW}/\n")

    for nazwa_pary, agenci in pary_agentow.items():
        config = buduj_config(agenci[0], agenci[1])
        zapisz_config(config)
        print(f"Para: {nazwa_pary}")

        for proba in range(1, ILOSC_POWTORZEN + 1):
            output_path = FOLDER_WYNIKOW / f"{nazwa_pary}_{proba}"
            print(f"  Próba {proba}/{ILOSC_POWTORZEN} → {output_path}.json / .txt", end=" ... ")

            ok = uruchom_probe(output_path, env)
            print("OK" if ok else "BŁĄD")

        print()

    print("Gotowe.")


if __name__ == "__main__":
    main()
