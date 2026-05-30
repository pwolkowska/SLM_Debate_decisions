"""
automatyzacja.py — Uruchamia eksperymenty dla wszystkich par/trójek agentów.

Dla każdej grupy agentów i każdej próby:
  1. Nadpisuje config.yaml
  2. Wywołuje main.py z --output wskazującym na docelowy plik
  3. main.py sam zapisuje <nazwa>.json i <nazwa>.txt

Wyniki lądują w folderze wyniki/:
    wyniki/
        otwartosc/
            O_wysoki-O_niski_1_seed42_v1.json
            ...
        trojki/
            O_wysoki-C_wysoki-N_niski_1_seed42_v1.json
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
SUFFIX = (
    "Bierzesz udział w dyskusji. "
    "Odpowiedz na argument rozmówcy. "
    "Pisz jednym akapitem, maksymalnie 4-5 zdań."
    "Kontynuuj dyskusję, nie podsumowuj."
)


# =====================================================================
# PARY AGENTÓW (2 agentów)
# =====================================================================
pary_agentow = {
    # ── Big Five (OCEAN) — high vs low ──────────────────────────────
    "O_wysoki-O_niski": [BIG5["O_wysoki"], BIG5["O_niski"]],
    "C_wysoki-C_niski": [BIG5["C_wysoki"], BIG5["C_niski"]],
    "E_wysoki-E_niski": [BIG5["E_wysoki"], BIG5["E_niski"]],
    "A_wysoki-A_niski": [BIG5["A_wysoki"], BIG5["A_niski"]],
    "N_wysoki-N_niski": [BIG5["N_wysoki"], BIG5["N_niski"]],
}

# =====================================================================
# TRÓJKI AGENTÓW (3 agentów)
# Klucz: dowolna nazwa; subfolder wyznaczany przez FOLDER_MAP lub "trojki"
# dla kluczy bez prefiksu OCEAN.
# =====================================================================
trojki_agentow = {
    # Przykład: jeden wysoki O, jeden wysoki C, jeden wysoki N
    # Odkomentuj lub dodaj własne trójki poniżej.

    # "O_wysoki-C_wysoki-N_wysoki": [BIG5["O_wysoki"], BIG5["C_wysoki"], BIG5["N_wysoki"]],
    # "E_wysoki-A_wysoki-N_niski":  [BIG5["E_wysoki"], BIG5["A_wysoki"], BIG5["N_niski"]],
}


# =====================================================================
# STAŁE USTAWIENIA
# =====================================================================
TOPIC = (
    "Firma ma kłopoty finansowe. Czy lepiej jest zwolnić 30% pracowników, "
    "żeby uratować pozostałych 70%, czy wszystkim obniżyć wypłatę o 20%, ale nikogo nie zwalniać?"
)
ILOSC_POWTORZEN = 4
SEEDS = [42, 256, 512, 1024]
FOLDER_WYNIKOW = Path("wyniki_eksperyment_1")


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def buduj_config(*agenci: dict, seed: int) -> dict:
    """Buduje config dla dowolnej liczby agentów (2, 3, ...)."""
    return {
        "model_name": "speakleash/Bielik-1.5B-v3.0-Instruct",
        "device": "cpu",
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
        # --- progi metryk ---
        # flip_threshold: obniżone z 0.20 → 0.13, bo Bielik na krótkich
        # wypowiedziach o tym samym temacie ma naturalnie niskie cos_dist
        # nawet przy semantycznym flip stanowiska
        "flip_threshold": 0.13,
        # soft-match dla claim_grounding / hedging / assertiveness:
        # obniżone z 0.45 → 0.32 (multilingual-MiniLM słabo separuje
        # krótkie markery przy wysokim progu)
        "claim_grounding_threshold": 0.32,
        "hedging_threshold": 0.32,
        "assertiveness_threshold": 0.32,
        # convergence_speed: zmiana z absolutnej na relatywną.
        # delta_threshold=0.03 → wykrywa faktyczny wzrost podobieństwa (~3pp),
        # co eliminuje fałszywe trafienia gdy sim jest wysoka już od R1 (np. cecha C)
        "convergence_speed_threshold": 0.92,   # zachowane dla reached_absolute
        "convergence_speed_delta": 0.03,        # nowe: próg relatywny (sim_t − sim_R1)
        "agents": [
            {"name": a["name"], "system_prompt": a["cecha"] + SUFFIX}
            for a in agenci
        ],
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


def get_trait_prefix(nazwa: str) -> str:
    """Zwraca literę cechy OCEAN (O/C/E/A/N) lub None dla mieszanych grup."""
    prefix = nazwa.split("_")[0]
    return prefix[0] if prefix[0] in FOLDER_MAP else None


def next_version_path(folder: Path, base_name: str) -> Path:
    i = 1
    while True:
        json_path = folder / f"{base_name}_v{i}.json"
        txt_path  = folder / f"{base_name}_v{i}.txt"
        if not json_path.exists() and not txt_path.exists():
            return folder / f"{base_name}_v{i}"
        i += 1


def uruchom_grupe(nazwa: str, agenci: list, subfolder: Path, env: dict):
    """Wspólna logika dla par i trójek."""
    print(f"Grupa: {nazwa}  ({len(agenci)} agentów)")

    for proba in range(1, ILOSC_POWTORZEN + 1):
        seed = SEEDS[proba - 1]
        config = buduj_config(*agenci, seed=seed)
        zapisz_config(config)

        subfolder.mkdir(parents=True, exist_ok=True)
        base_name = f"{nazwa}_{proba}_seed{seed}"
        output_path = next_version_path(subfolder, base_name)
        print(
            f"  Próba {proba}/{ILOSC_POWTORZEN}  seed={seed} → {output_path}.json / .txt",
            end=" ... ",
        )

        ok = uruchom_probe(output_path, env)
        print("OK" if ok else "BŁĄD")

    print()


# =====================================================================
# GŁÓWNA PĘTLA
# =====================================================================

def main():
    FOLDER_WYNIKOW.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    print(f"Rozpoczynam eksperymenty. Wyniki → {FOLDER_WYNIKOW}/\n")

    # --- Pary ---
    for nazwa, agenci in pary_agentow.items():
        trait = get_trait_prefix(nazwa)
        subfolder = FOLDER_WYNIKOW / (FOLDER_MAP[trait] if trait else "mieszane")
        uruchom_grupe(nazwa, agenci, subfolder, env)

    # --- Trójki ---
    for nazwa, agenci in trojki_agentow.items():
        trait = get_trait_prefix(nazwa)
        subfolder = FOLDER_WYNIKOW / (FOLDER_MAP[trait] if trait else "trojki")
        uruchom_grupe(nazwa, agenci, subfolder, env)

    print("Gotowe.")


if __name__ == "__main__":
    main()
