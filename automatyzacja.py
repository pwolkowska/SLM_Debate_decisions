"""
automatyzacja.py — Uruchamia eksperymenty dla wszystkich par lub trójek agentów.

Użycie:
    python automatyzacja.py --agenci 2   # pary (domyślnie)
    python automatyzacja.py --agenci 3   # trójki
"""

import argparse
import os
import subprocess
import yaml
from pathlib import Path

from osobowosci import BIG5

# =====================================================================
# SUFFIX — wspólny dla wszystkich agentów
# =====================================================================
SUFFIX = (
    " Bierzesz udział w debacie jako jeden z rozmówców. "
    "Wypowiadasz się WYŁĄCZNIE we własnym imieniu. "
    "NIE wcielaj się w inne osoby. NIE pisz 'Osoba 1:', 'Osoba 2:' ani żadnych etykiet. "
    "NIE przepisuj ani NIE streszczaj poprzednich wypowiedzi. "
    "Zacznij od razu od swojego argumentu lub reakcji na poprzednika. "
    "Pisz w 2-3 zdaniach, bez list, bez numeracji, bez gwiazdek, bez pogrubień. "
    "Czysty tekst, jakbyś mówił do rozmówcy twarzą w twarz. "
)

# =====================================================================
# PARY AGENTÓW (2 agentów)
# =====================================================================
pary_agentow = {
    "emocjonalny-racjonalny": [
        {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
        {"name": "Racjonalny",  "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    ],
    "racjonalny-emocjonalny": [
        {"name": "Racjonalny",  "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
        {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
    ],
    # "konfrontacyjny-unikajacy": [
    #     {"name": "Konfrontacyjny",      "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie. Mówisz stanowczo i nie boisz się napięcia w rozmowie."},
    #     {"name": "Unikajacy konfliktu", "cecha": "Jesteś nastawiony na unikanie konfliktów. Starasz się łagodzić napięcia i utrzymywać dobrą atmosferę w rozmowie."},
    # ],
    # "unikajacy-konfrontacyjny": [
    #     {"name": "Unikajacy konfliktu", "cecha": "Jesteś nastawiony na unikanie konfliktów. Starasz się łagodzić napięcia i utrzymywać dobrą atmosferę w rozmowie."},
    #     {"name": "Konfrontacyjny",      "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie. Mówisz stanowczo i nie boisz się napięcia w rozmowie."},
    # ],
    # "bezwzgledny-ustepliwy": [
    #     {"name": "Bezwzgledny", "cecha": "Jesteś osobą bezwzględną. Dążysz do realizacji swoich celów za wszelką cenę, nawet jeśli oznacza to poświęcenie innych."},
    #     {"name": "Ustepliwy",   "cecha": "Jesteś osobą ustępliwą. Cenisz harmonię i dobre relacje z innymi, nawet jeśli oznacza to rezygnację z własnych celów."},
    # ],
    # "bezwzgledny-empata": [
    #     {"name": "Bezwzgledny", "cecha": "Jesteś osobą bezwzględną, zdecydowaną i nastawioną na drastyczne kroki."},
    #     {"name": "Empata",      "cecha": "Jesteś osobą wyjątkowo wrażliwą, opiekuńczą i współczującą."},
    # ],
    # "empata-bezwzgledny": [
    #     {"name": "Empata",      "cecha": "Jesteś osobą wyjątkowo wrażliwą, opiekuńczą i współczującą."},
    #     {"name": "Bezwzgledny", "cecha": "Jesteś osobą bezwzględną, zdecydowaną i nastawioną na drastyczne kroki."},
    # ],
    # "asekurant-ryzykant": [
    #     {"name": "Asekurant", "cecha": "Jesteś osobą lękliwą, skrajnie ostrożną i nieufną wobec nowości."},
    #     {"name": "Ryzykant",  "cecha": "Jesteś osobą bardzo odważną i czerpiesz satysfakcję z podejmowania niebezpiecznych, ekstremalnych decyzji."},
    # ],
    # "ryzykant-asekurant": [
    #     {"name": "Ryzykant",  "cecha": "Jesteś osobą bardzo odważną i czerpiesz satysfakcję z podejmowania niebezpiecznych, ekstremalnych decyzji."},
    #     {"name": "Asekurant", "cecha": "Jesteś osobą lękliwą, skrajnie ostrożną i nieufną wobec nowości."},
    # ],
    # "optymista-pesymista": [
    #     {"name": "Optymista", "cecha": "Jesteś pozytywnie nastawiony do świata i ludzi. Wierzysz, że nawet trudne sytuacje mają w sobie potencjał do poprawy. Unikasz czarnych scenariuszy."},
    #     {"name": "Pesymista", "cecha": "Jesteś osobą ostrożną i sceptyczną. Koncentrujesz się na tym, co może pójść nie tak. Masz skłonność do analizowania najgorszych możliwych scenariuszy i traktujesz je jako punkt wyjścia do oceny sytuacji."},
    # ],
    # "pesymista-optymista": [
    #     {"name": "Pesymista", "cecha": "Jesteś osobą ostrożną i sceptyczną. Koncentrujesz się na tym, co może pójść nie tak. Masz skłonność do analizowania najgorszych możliwych scenariuszy i traktujesz je jako punkt wyjścia do oceny sytuacji."},
    #     {"name": "Optymista", "cecha": "Jesteś pozytywnie nastawiony do świata i ludzi. Wierzysz, że nawet trudne sytuacje mają w sobie potencjał do poprawy. Unikasz czarnych scenariuszy."},
    # ],

    # ── Big Five (OCEAN) — high vs low ──────────────────────────────
    "O_wysoki-O_niski": [BIG5["O_wysoki"], BIG5["O_niski"]],
    # "C_wysoki-C_niski": [BIG5["C_wysoki"], BIG5["C_niski"]],
    # "E_wysoki-E_niski": [BIG5["E_wysoki"], BIG5["E_niski"]],
    # "A_wysoki-A_niski": [BIG5["A_wysoki"], BIG5["A_niski"]],
    # "N_wysoki-N_niski": [BIG5["N_wysoki"], BIG5["N_niski"]],
}

# =====================================================================
# TRÓJKI AGENTÓW (3 agentów)
# =====================================================================
trojki_agentow = {
    # ── Aktywne ─────────────────────────────────────────────────────
    "emocjonalny-racjonalny-ugodowy": [
        {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
        {"name": "Racjonalny",  "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
        {"name": "Ugodowy",     "cecha": "Jesteś osobą życzliwą i nastawioną na współpracę. Szukasz rozwiązania, które zaakceptują wszyscy."},
    ],

    # ── Big Five — mieszane wymiary ──────────────────────────────────
    # Otwarty + Konwencjonalny + Ugodowy (kreator vs tradycjonalista + mediator)
    # "O_wysoki-O_niski-A_wysoki": [BIG5["O_wysoki"], BIG5["O_niski"], BIG5["A_wysoki"]],

    # Otwarty + Sumienny + Ugodowy ("dream team" — różne style, wszyscy konstruktywni)
    # "O_wysoki-C_wysoki-A_wysoki": [BIG5["O_wysoki"], BIG5["C_wysoki"], BIG5["A_wysoki"]],

    # Otwarty + Sumienny + Niezależny (kreator vs planista vs indywidualista)
    # "O_wysoki-C_wysoki-A_niski": [BIG5["O_wysoki"], BIG5["C_wysoki"], BIG5["A_niski"]],

    # Ekstrawertyk + Introwertyk + Ugodowy (dominujący + milczący + łagodzący)
    # "E_wysoki-E_niski-A_wysoki": [BIG5["E_wysoki"], BIG5["E_niski"], BIG5["A_wysoki"]],

    # Ekstrawertyk + Niezależny + Wrażliwy (asertywny + uparty + lękliwy — konfliktowa trójka)
    # "E_wysoki-A_niski-N_wysoki": [BIG5["E_wysoki"], BIG5["A_niski"], BIG5["N_wysoki"]],

    # Wrażliwy + Stabilny + Ugodowy (lękliwy vs spokojny + mediator — test wpływu neurotyczności)
    # "N_wysoki-N_niski-A_wysoki": [BIG5["N_wysoki"], BIG5["N_niski"], BIG5["A_wysoki"]],

    # Wrażliwy + Stabilny + Niezależny (lękliwy vs spokojny + uparty)
    # "N_wysoki-N_niski-A_niski": [BIG5["N_wysoki"], BIG5["N_niski"], BIG5["A_niski"]],

    # Otwarty + Spontaniczny + Wrażliwy (chaos twórczy — brak stabilności)
    # "O_wysoki-C_niski-N_wysoki": [BIG5["O_wysoki"], BIG5["C_niski"], BIG5["N_wysoki"]],

    # Sumienny + Spontaniczny + Ekstrawertyk (zorganizowany vs chaotyczny + towarzyski)
    # "C_wysoki-C_niski-E_wysoki": [BIG5["C_wysoki"], BIG5["C_niski"], BIG5["E_wysoki"]],

    # Sumienny + Niezależny + Wrażliwy (odpowiedzialny + uparty + lękliwy)
    # "C_wysoki-A_niski-N_wysoki": [BIG5["C_wysoki"], BIG5["A_niski"], BIG5["N_wysoki"]],

    # ── Custom — tematyczne trójki ───────────────────────────────────
    # Klasyczny konflikt + mediator
    # "optymista-pesymista-ugodowy": [
    #     {"name": "Optymista", "cecha": "Jesteś pozytywnie nastawiony do świata. Wierzysz, że nawet trudne sytuacje mają potencjał do poprawy."},
    #     {"name": "Pesymista", "cecha": "Jesteś ostrożny i sceptyczny. Koncentrujesz się na tym, co może pójść nie tak."},
    #     {"name": "Ugodowy",   "cecha": "Jesteś życzliwy i nastawiony na współpracę. Szukasz rozwiązania, które zaakceptują wszyscy."},
    # ],

    # Ryzyko + ostrożność + rozsądek
    # "ryzykant-asekurant-racjonalny": [
    #     {"name": "Ryzykant",   "cecha": "Jesteś bardzo odważny i czerpiesz satysfakcję z podejmowania ekstremalnych decyzji."},
    #     {"name": "Asekurant",  "cecha": "Jesteś lękliwy, skrajnie ostrożny i nieufny wobec nowości."},
    #     {"name": "Racjonalny", "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    # ],

    # Emocje + logika + konflikt
    # "emocjonalny-racjonalny-konfrontacyjny": [
    #     {"name": "Emocjonalny",     "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
    #     {"name": "Racjonalny",      "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    #     {"name": "Konfrontacyjny",  "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie i nie boisz się napięcia."},
    # ],

    # Trójka bez kompromisu — wszyscy uparci
    # "racjonalny-niezalezny-konfrontacyjny": [
    #     {"name": "Racjonalny",     "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
    #     {"name": "Niezależny",     "cecha": "Jesteś osobą bezpośrednią i niezależną. Bronisz swojego stanowiska i nie poddajesz się presji."},
    #     {"name": "Konfrontacyjny", "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie i nie boisz się napięcia."},
    # ],
}

# =====================================================================
# STAŁE USTAWIENIA
# =====================================================================
TOPIC = (
    "Firma ma kłopoty finansowe. Czy lepiej jest zwolnić 30% pracowników, "
    "żeby uratować pozostałych 70%, czy wszystkim obniżyć wypłatę o 20%, ale nikogo nie zwalniać?"
)
ILOSC_POWTORZEN = 1
SEEDS = [42, 137, 256, 512, 1024]
FOLDER_WYNIKOW = Path("wyniki")


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def buduj_config(agenci: list, seed: int) -> dict:
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
        "agents": [
            {"name": a["name"], "system_prompt": a["cecha"] + SUFFIX}
            for a in agenci
        ]
    }


def zapisz_config(config: dict):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def uruchom_probe(output_path: Path, env: dict) -> bool:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agenci", type=int, choices=[2, 3], default=2,
        help="Liczba agentów: 2 (pary) lub 3 (trójki). Domyślnie: 2"
    )
    args = parser.parse_args()

    FOLDER_WYNIKOW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    eksperymenty = pary_agentow if args.agenci == 2 else trojki_agentow
    print(f"Tryb: {args.agenci} agentów | Wyniki → {FOLDER_WYNIKOW}/\n")

    for nazwa, agenci in eksperymenty.items():
        print(f"Grupa: {nazwa}")
        for proba in range(1, ILOSC_POWTORZEN + 1):
            seed = SEEDS[proba - 1]
            config = buduj_config(agenci, seed)
            zapisz_config(config)
            output_path = FOLDER_WYNIKOW / f"{nazwa}_{proba}_seed{seed}"
            print(f"  Próba {proba}/{ILOSC_POWTORZEN} seed={seed} → {output_path}", end=" ... ")
            ok = uruchom_probe(output_path, env)
            print("OK" if ok else "BŁĄD")
        print()

    print("Gotowe.")


if __name__ == "__main__":
    main()