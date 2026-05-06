import subprocess
import yaml
import os

SUFFIX_AGENT_1 = " Zanim wyrazisz swoje zdanie, zapoznaj się z wypowiedzią drugiego rozmówcy. Jeśli historia jest pusta, to znaczy, że zaczynasz dyskusję - wówczas przedstaw swoj punkt widzenia w sposób możliwie przekonujący dla Twojego rozmówcy. Waszym celem jest możliwie szybko wspólnie znaleźć najlepsze rozwiązanie zadanego tematu."

SUFFIX_AGENT_2 = " Prowadzisz rozmowę i szukasz wspólnej odpowiedzi. Zanim wyrazisz swoje zdanie, zapoznaj się z wypowiedzią drugiego rozmówcy. Waszym celem jest możliwie szybko znaleźć najlepsze rozwiązanie zadanego tematu."

# =====================================================================
# 1. PARY AGENTÓW I NAZWY PLIKÓW
# =====================================================================
# do bazowej nazwy pliku skrypt sam dopisze 1.txt i 2.txt.
pary_agentow = {
    
    "ryzykant-asekurant": [
        {"name": "Ryzykant", "cecha": "Jesteś osobą bardzo odważną i czerpiesz satysfakcję z podejmowania niebezpiecznych, ekstremalnych decyzji."},
        {"name": "Asekurant", "cecha": "Jesteś osobą lękliwą, skrajnie ostrożną i nieufną wobec nowości."}
    ],

    "pesymista-optymista": [
        {"name": "Pesymista", "cecha": "Jesteś osobą ostrożną i sceptyczną. Koncentrujesz się na tym, co może pójść nie tak. Masz skłonność do analizowania najgorszych możliwych scenariuszy i traktujesz je jako punkt wyjścia do oceny sytuacji."},
        {"name": "Optymista", "cecha": "Jesteś  pozytywnie nastawiony do świata i ludzi. Wierzysz, że nawet trudne sytuacje mają w sobie potencjał do poprawy. Unikasz czarnych scenariuszy."}
    ],

     "emocjonalny-racjonalny": [
        {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."},
        {"name": "Racjonalny", "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."}
    ],

     "racjonalny-emocjonalny": [
        {"name": "Racjonalny", "cecha": "Jesteś osobą racjonalną. Kierujesz się twardymi faktami i logiką."},
        {"name": "Emocjonalny", "cecha": "Jesteś osobą emocjonalną. Kierujesz się empatią."}
    ],

    "konfrontacyjny-unikajacy_konfliktu": [
        {"name": "Konfrontacyjny", "cecha": "Jesteś osobą bezpośrednią i konfrontacyjną. Otwarcie wyrażasz swoje zdanie. Mówisz stanowczo i nie boisz się napięcia w rozmowie."},
        {"name": "Unikający konfliktu", "cecha": "Jesteś nastawiony na  unikanie konfliktów. Starasz się łagodzić napięcia i utrzymywać dobrą atmosferę w rozmowie.."}
    ],

}

# =====================================================================
# 2. STAŁE USTAWIENIA
# =====================================================================
topic = "Firma ma kłopoty finansowe. Czy lepiej jest zwolnić 30% pracowników, żeby uratować pozostałych 70%, czy wszystkim obniżyć wypłatę o 20%, ale nikogo nie zwalniać?"
ilosc_powtorzen = 2

# Skrypt upewnia się, że folder "wyniki" istnieje
folder_wynikow = "wyniki"
os.makedirs(folder_wynikow, exist_ok=True)

def zapisz_config(agent1, agent2):
    """Funkcja łącząca cechy z suffixami i nadpisująca plik config.yaml"""
    
    # Sklejanie ostatecznych promptów: cecha charakteru + wspólna końcówka
    prompt_agent_1 = agent1["cecha"] + SUFFIX_AGENT_1
    prompt_agent_2 = agent2["cecha"] + SUFFIX_AGENT_2
    
    config = {
        "model_name": "speakleash/Bielik-1.5B-v3.0-Instruct",
        "device": "cpu",
        "temperature": 0.7,
        "max_new_tokens": 256,
        "do_sample": True,
        "architecture": "round_robin",
        "num_rounds": 3,
        "topic": topic,
        "decision_protocol": "consensus",
        "consensus_threshold": 1.0,
        "max_consensus_rounds": 3,
        "agents": [
            {"name": agent1["name"], "system_prompt": prompt_agent_1},
            {"name": agent2["name"], "system_prompt": prompt_agent_2}
        ],
        "judge": {
            "system_prompt": "Jesteś sędzią. (Ten prompt i tak jest ignorowany przy konsensusie)."
        }
    }
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

# =====================================================================
# 4. GŁÓWNA PĘTLA URUCHAMIAJĄCA
# =====================================================================
print(f"Rozpoczynam automatyzację testów. Pliki wylądują w folderze '{folder_wynikow}'...\n")

env_vars = os.environ.copy()
env_vars["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

for bazowa_nazwa, agenci in pary_agentow.items():
    print(f"--- Przygotowuję testy dla bazowej nazwy: {bazowa_nazwa} ---")
    
    # Przekazujemy obu agentów do funkcji łączącej teksty (index 0 to Agent 1, index 1 to Agent 2)
    zapisz_config(agenci[0], agenci[1])

    for proba in range(1, ilosc_powtorzen + 1):
        nazwa_pliku = f"{bazowa_nazwa}_{proba}.txt"
        pelna_sciezka = os.path.join(folder_wynikow, nazwa_pliku)
        
        print(f" -> Uruchamiam próbę {proba}/{ilosc_powtorzen}... Zapisuję do: {pelna_sciezka}")

        with open(pelna_sciezka, "w", encoding="utf-8") as plik_wynikowy:
            subprocess.run(
                ["python", "-X", "utf8", "main.py"],
                stdout=plik_wynikowy,
                stderr=subprocess.STDOUT,  
                text=True,
                encoding="utf-8",
                env=env_vars
            )
        
print("\nGotowe! Wszystkie testy zostały pomyślnie wygenerowane.")