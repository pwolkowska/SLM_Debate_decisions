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
import sys
import yaml
from pathlib import Path

from osobowosci import BIG5

FOLDER_MAP = {
    "O": "otwartosc",
    "C": "sumiennosc",
    "E": "ekstrawersja",
    "A": "ugodowosc",
    "N": "neurotycznosc",
}

# =====================================================================
# SUFFIX Promptu
# =====================================================================
# SUFFIX_AGENT_1 = (
#     # " Bierzesz udział w debacie. "
#     # "W pierwszej wypowiedzi jasno powiedz, którą opcję uważasz za lepszą i dlaczego — na podstawie swoich wartości i charakteru. "
#     # "Broń swojego zdania. "
#     # "Zwróć uwagę na argumenty przeciwnika i odpowiedz na nie. "
#     # "Odpowiadaj w maksymalnie 3-4 zdaniach. Nie formatuj odpowiedzi, zwracaj czysty tekst bez numeracji, punktorów czy pogrubień. "
#     "Bierzesz udział w debacie. "
#     "Przedstaw swoje stanowisko i je uzasadnij. "
#     "Odnoś się do argumentów drugiej strony. "
#     "Odpowiadaj krótko, maksymalnie 3-4 zdania. "
#     "Zwracaj wyłącznie czysty tekst."
# )

# SUFFIX_AGENT_2 = (
#     # " Bierzesz udział w debacie. "
#     # "W pierwszej wypowiedzi jasno powiedz, którą opcję uważasz za lepszą i dlaczego — na podstawie swoich wartości i charakteru. "
#     # "Broń swojego zdania. "
#     # "Zwróć uwagę na argumenty przeciwnika i odpowiedz na nie. "
#     # "Odpowiadaj w maksymalnie 3-4 zdaniach. Nie formatuj odpowiedzi, zwracaj czysty tekst bez numeracji, punktorów czy pogrubień. "
#     "Bierzesz udział w debacie. "
#     "Przedstaw swoje stanowisko i je uzasadnij. "
#     "Odnoś się do argumentów drugiej strony. "
#     "Odpowiadaj krótko, maksymalnie 3-4 zdania. "
#     "Zwracaj wyłącznie czysty tekst."
# )
SUFFIX = (
    # " Bierzesz udział w debacie. "
    # "W pierwszej wypowiedzi jasno powiedz, którą opcję uważasz za lepszą i dlaczego — na podstawie swoich wartości i charakteru. "
    # "Broń swojego zdania. "
    # "Zwróć uwagę na argumenty przeciwnika i odpowiedz na nie. "
    # "Odpowiadaj w maksymalnie 3-4 zdaniach. Nie formatuj odpowiedzi, zwracaj czysty tekst bez numeracji, punktorów czy pogrubień. "
    
    # "Bierzesz udział w dyskusji. "
    # "Powiedz którą opcję wybierasz i dlaczego. "
    # "Odpowiedz na to co powiedział rozmówca. "
    # "Pisz jednym ciągłym akapitem, nie używaj myślników, cyfr ani gwiazdek. "
    # "Maksymalnie 3-4 zdania."

    "Bierzesz udział w dyskusji. "
    "Odpowiedz na argument rozmówcy. "
    "Pisz jednym akapitem, maksymalnie 4-5 zdań."
    "Kontynuuj dyskusję, nie podsumowuj."
)



# =====================================================================
# PARY AGENTÓW
# =====================================================================
pary_agentow = {
    "O_wysoki-O_niski": [BIG5["O_wysoki"], BIG5["O_niski"]],
    "C_wysoki-C_niski": [BIG5["C_wysoki"], BIG5["C_niski"]],
    "E_wysoki-E_niski": [BIG5["E_wysoki"], BIG5["E_niski"]],
    "A_wysoki-A_niski": [BIG5["A_wysoki"], BIG5["A_niski"]],
    "N_wysoki-N_niski": [BIG5["N_wysoki"], BIG5["N_niski"]],
}

# =====================================================================
# STAŁE USTAWIENIA
# =====================================================================
TOPIC = (
    "Firma ma kłopoty finansowe. Czy lepiej jest zwolnić 30% pracowników, "
    "żeby uratować pozostałych 70%, czy wszystkim obniżyć wypłatę o 20%, ale nikogo nie zwalniać?"
)
SEEDS = [42, 256, 512, 1024]
ILOSC_POWTORZEN = 4  # ile seedów użyć
FOLDER_WYNIKOW = Path("wyniki_eksperyment_1")


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def buduj_config(agent1: dict, agent2: dict, seed: int) -> dict:
    return {
        "model_name": "speakleash/Bielik-1.5B-v3.0-Instruct",
        "device": "cuda",
        "temperature": 0.7,
        "max_new_tokens": 200,
        "do_sample": True,
        "seed": seed,
        "architecture": "round_robin",
        "num_rounds": 3,
        "topic": TOPIC,
        "decision_protocol": "consensus",
        "consensus_threshold": 1.0,
        "max_consensus_rounds": 8,
        "agents": [
            {"name": agent1["name"], "system_prompt": agent1["cecha"] + SUFFIX},
            {"name": agent2["name"], "system_prompt": agent2["cecha"] + SUFFIX},
        ]
    }


def zapisz_config(config: dict):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def uruchom_probe(output_path: Path, env: dict) -> bool:
    """Uruchamia main.py z --output. Zwraca True jeśli sukces."""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "main.py", "--output", str(output_path)],
        text=True,
        encoding="utf-8",
        env=env,
    )
    return result.returncode == 0

def get_trait_prefix(nazwa_pary: str) -> str:
    return nazwa_pary.split("_")[0][0]  # O, C, E, A, N

def next_version_path(folder: Path, base_name: str) -> Path:
    i = 1
    while True:
        json_path = folder / f"{base_name}_v{i}.json"
        txt_path = folder / f"{base_name}_v{i}.txt"

        if not json_path.exists() and not txt_path.exists():
            return folder / f"{base_name}_v{i}"

        i += 1


# =====================================================================
# GŁÓWNA PĘTLA
# =====================================================================

def main():
    FOLDER_WYNIKOW.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    print(f"Rozpoczynam eksperymenty. Wyniki → {FOLDER_WYNIKOW}/\n")

    for nazwa_pary, agenci in pary_agentow.items():
        print(f"Para: {nazwa_pary}")

        for proba, seed in enumerate(SEEDS[:ILOSC_POWTORZEN], 1):
            config = buduj_config(agenci[0], agenci[1], seed)
            zapisz_config(config)

            trait = get_trait_prefix(nazwa_pary)
            subfolder = FOLDER_WYNIKOW / FOLDER_MAP[trait]
            subfolder.mkdir(parents=True, exist_ok=True)

            base_name = f"{nazwa_pary}_{proba}_seed{seed}"
            output_path = next_version_path(subfolder, base_name)
            print(f"  Próba {proba}/{ILOSC_POWTORZEN} seed={seed} → {output_path}.json / .txt", end=" ... ")

            ok = uruchom_probe(output_path, env)
            print("OK" if ok else "BŁĄD")

        print()

    print("Gotowe.")


if __name__ == "__main__":
    main()
