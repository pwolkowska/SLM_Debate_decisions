"""
metrics.py — Obliczanie metryk z ustrukturyzowanych danych debaty.

5 metryk, każda odpowiada na jedno pytanie badawcze:
    tokens_per_turn          — czy osobowość wpływa na ilość produkowanego tekstu?
    opinion_shift            — czy agenci zmieniają stanowisko w toku debaty?
    between_agent_similarity — czy agenci produkują różne argumenty i czy się zbliżają?
    lexical_richness         — czy osobowość wpływa na różnorodność słownictwa?
    convergence              — czy debata kończy się porozumieniem i jak szybko?

Metryki opinion_shift i between_agent_similarity wymagają: pip install sentence-transformers
"""

import re
from collections import defaultdict


# =============================================================================
# Funkcje pomocnicze
# =============================================================================

def _mean(values):
    return round(sum(values) / len(values), 4) if values else None

def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 4)

def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return round((sum((x - m) ** 2 for x in values) / len(values)) ** 0.5, 4)

def _load_embeddings(texts: list):
    """Zwraca (embeddings_array, numpy_module) lub (None, None) jeśli brak biblioteki lub połączenia."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
        emb = model.encode(texts, normalize_embeddings=True)
        return emb, np
    except Exception:
        return None, None


# =============================================================================
# METRYKI
# =============================================================================

def tokens_per_turn(debate_log: list) -> dict:
    """Verbosity per agent i overall. Nie wymaga embeddingów."""
    all_tokens = [e["tokens"] for e in debate_log]

    by_agent = defaultdict(list)
    for e in debate_log:
        by_agent[e["agent"]].append(e["tokens"])

    per_agent = {}
    for agent, tokens in by_agent.items():
        per_agent[agent] = {
            "mean": _mean(tokens),
            "median": _median(tokens),
            "std": _std(tokens),
            "total": sum(tokens),
        }

    return {
        "overall": {
            "mean": _mean(all_tokens),
            "median": _median(all_tokens),
            "std": _std(all_tokens),
            "total": sum(all_tokens),
        },
        "per_agent": per_agent,
    }


def opinion_shift(debate_log: list, embeddings, np) -> dict:
    """Semantyczna zmiana stanowiska: 1 - cosine_sim(pierwsza, ostatnia wypowiedź) per agent.

    0.0 = agent nie zmienił stanowiska, 1.0 = całkowita zmiana.
    """
    if embeddings is None:
        return {"per_agent": {}, "mean": None, "note": "brak sentence-transformers"}

    by_agent = defaultdict(list)
    for idx, e in enumerate(debate_log):
        by_agent[e["agent"]].append(idx)

    per_agent = {}
    for agent, idxs in by_agent.items():
        if len(idxs) < 2:
            per_agent[agent] = None
            continue
        first, last = idxs[0], idxs[-1]
        cosine_sim = float(np.dot(embeddings[first], embeddings[last]))
        per_agent[agent] = round(1.0 - cosine_sim, 4)

    valid = [v for v in per_agent.values() if v is not None]
    return {
        "per_agent": per_agent,
        "mean": _mean(valid),
    }


def between_agent_similarity(debate_log: list, embeddings, np) -> dict:
    """Podobieństwo semantyczne między agentami per runda.

    Blisko 1.0 = agenci mówią to samo, blisko 0.0 = całkowicie różne argumenty.
    delta = final - initial: >0 oznacza zbliżanie poglądów, <0 = oddalanie.
    """
    if embeddings is None:
        return {"per_round": {}, "initial": None, "final": None, "delta": None,
                "note": "brak sentence-transformers"}

    by_round = defaultdict(list)
    for idx, e in enumerate(debate_log):
        by_round[e["round"]].append(idx)

    per_round = {}
    for rnd, idxs in sorted(by_round.items()):
        if len(idxs) < 2:
            per_round[rnd] = None
            continue
        pairs = [
            float(np.dot(embeddings[i], embeddings[j]))
            for ii, i in enumerate(idxs)
            for j in idxs[ii + 1:]
        ]
        per_round[rnd] = round(_mean(pairs), 4) if pairs else None

    valid_rounds = {r: v for r, v in per_round.items() if v is not None}
    if not valid_rounds:
        return {"per_round": per_round, "initial": None, "final": None, "delta": None}

    sorted_rounds = sorted(valid_rounds.keys())
    initial = valid_rounds[sorted_rounds[0]]
    final = valid_rounds[sorted_rounds[-1]]

    return {
        "per_round": per_round,
        "initial": initial,
        "final": final,
        "delta": round(final - initial, 4),
    }


def lexical_richness(debate_log: list) -> dict:
    """Type-Token Ratio per agent: unikalne_słowa / wszystkie_słowa.

    Wyższe TTR = bogatsze, bardziej różnorodne słownictwo.
    Nie wymaga embeddingów.
    """
    by_agent = defaultdict(list)
    for e in debate_log:
        by_agent[e["agent"]].append(e["text"])

    per_agent = {}
    for agent, texts in by_agent.items():
        combined = " ".join(texts).lower()
        words = re.findall(r"\b[a-ząćęłńóśźż]+\b", combined)
        total = len(words)
        unique = len(set(words))
        per_agent[agent] = {
            "ttr": round(unique / total, 4) if total > 0 else None,
            "unique_words": unique,
            "total_words": total,
        }

    return {"per_agent": per_agent}


def convergence(decision_result: dict) -> dict:
    """Wynik protokołu consensus: czy osiągnięto porozumienie i jak szybko."""
    return {
        "reached": decision_result.get("consensus_reached", False),
        "convergence_round": decision_result.get("convergence_round", None),
        "agreement_ratios": [
            r["agreement_ratio"]
            for r in decision_result.get("consensus_rounds", [])
        ],
    }


# =============================================================================
# Funkcja zbiorcza — wywołana przez main.py
# =============================================================================

def compute_all(debate_log: list, decision_result: dict, config: dict) -> dict:
    """Liczy 5 metryk i zwraca jeden słownik. Embeddingi ładowane raz."""
    texts = [e["text"] for e in debate_log]
    embeddings, np = _load_embeddings(texts)

    return {
        "tokens_per_turn": tokens_per_turn(debate_log),
        "opinion_shift": opinion_shift(debate_log, embeddings, np),
        "between_agent_similarity": between_agent_similarity(debate_log, embeddings, np),
        "lexical_richness": lexical_richness(debate_log),
        "convergence": convergence(decision_result),
    }
