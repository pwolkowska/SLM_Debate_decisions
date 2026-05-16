"""
metrics.py — Obliczanie metryk z ustrukturyzowanych danych debaty.

Importowany przez main.py — nie ma własnego CLI ani parsera plików tekstowych.
Wszystkie funkcje operują na debate_log i decision_result przekazanych wprost.

Etap 1 (zawsze dostępne, zero dodatkowych bibliotek):
    tokens_per_turn       #5  — długość wypowiedzi per agent
    flip_rate             #3  — NoF i ToF per agent (proxy Jaccard)
    entropy               #11 — entropia rozkładu odpowiedzi per runda
    convergence_round     #1  — pierwsza runda konsensusu (z danych lub proxy)
    auc_agreement         #2  — średnia frakcja zgody z większością po rundach

Etap 2 (wymaga: pip install sentence-transformers):
    semantic_diversity    #8  — 1 − mean(cosine_similarity) między embeddingami
    redundancy_ratio      #15 — frakcja wypowiedzi bardzo podobna do wcześniejszych
"""

import math
from collections import Counter


# =============================================================================
# ETAP 1 — bez dodatkowych bibliotek
# =============================================================================

def tokens_per_turn(debate_log: list) -> dict:
    """#5 Średnia/mediana/std liczby tokenów per wypowiedź."""
    all_tokens = [e["tokens"] for e in debate_log]
    per_agent = {}
    for e in debate_log:
        per_agent.setdefault(e["agent"], []).append(e["tokens"])

    return {
        "overall": {
            "mean": _mean(all_tokens),
            "median": _median(all_tokens),
            "std": _std(all_tokens),
            "total": sum(all_tokens),
        },
        "per_agent": {
            agent: {
                "mean": _mean(vals),
                "median": _median(vals),
                "std": _std(vals),
            }
            for agent, vals in per_agent.items()
        },
    }


def flip_rate(debate_log: list) -> dict:
    """#3 NoF (liczba zmian) i ToF (runda pierwszej zmiany) per agent.

    Zmiana = Jaccard similarity między kolejnymi wypowiedziami tego samego agenta < 0.6.
    """
    by_agent = {}
    for e in debate_log:
        by_agent.setdefault(e["agent"], []).append(e)

    result = {}
    for agent, entries in by_agent.items():
        entries_sorted = sorted(entries, key=lambda x: x["round"])
        flips = []
        for i in range(1, len(entries_sorted)):
            prev = set(entries_sorted[i - 1]["text"].lower().split())
            curr = set(entries_sorted[i]["text"].lower().split())
            union = len(prev | curr)
            jaccard = len(prev & curr) / union if union else 1.0
            if jaccard < 0.6:
                flips.append(entries_sorted[i]["round"])

        result[agent] = {
            "NoF": len(flips),
            "ToF": flips[0] if flips else None,
            "flip_rounds": flips,
        }

    return result


def entropy(debate_log: list) -> dict:
    """#11 Entropia Shannona rozkładu odpowiedzi per runda."""
    by_round = {}
    for e in debate_log:
        by_round.setdefault(e["round"], []).append(e["text"][:60].lower().strip())

    per_round = {}
    for rnd, snippets in by_round.items():
        counts = Counter(snippets)
        total = sum(counts.values())
        H = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
        per_round[rnd] = round(H, 4)

    values = list(per_round.values())
    return {
        "per_round": per_round,
        "mean": round(_mean(values), 4) if values else None,
    }


def convergence_round(decision_result: dict, debate_log: list, threshold: float = 0.5) -> dict:
    """#1 Pierwsza runda, w której osiągnięto konsensus."""
    if decision_result.get("protocol") == "consensus":
        return {
            "convergence_round": decision_result.get("convergence_round"),
            "reached": decision_result.get("consensus_reached", False),
            "method": "consensus_protocol",
        }

    # Proxy przez Jaccard similarity wypowiedzi w rundzie
    by_round = {}
    for e in debate_log:
        by_round.setdefault(e["round"], []).append(set(e["text"].lower().split()))

    for rnd in sorted(by_round.keys()):
        word_sets = by_round[rnd]
        if len(word_sets) < 2:
            continue
        sims = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                u = len(word_sets[i] | word_sets[j])
                sims.append(len(word_sets[i] & word_sets[j]) / u if u else 0)
        if _mean(sims) >= 0.4:
            return {
                "convergence_round": rnd,
                "reached": True,
                "method": "jaccard_proxy",
                "avg_similarity": round(_mean(sims), 4),
            }

    return {"convergence_round": None, "reached": False, "method": "jaccard_proxy"}


def auc_agreement(debate_log: list) -> dict:
    """#2 AUC-Agreement — średnia frakcja agentów zgadzających się z większością per runda."""
    by_round = {}
    for e in debate_log:
        by_round.setdefault(e["round"], []).append(e)

    per_round = {}
    for rnd, entries in by_round.items():
        snippets = [e["text"][:60].lower().strip() for e in entries]
        majority = Counter(snippets).most_common(1)[0][0]
        agree_count = sum(1 for s in snippets if s == majority)
        per_round[rnd] = round(agree_count / len(snippets), 4)

    values = list(per_round.values())
    return {
        "per_round": per_round,
        "auc": round(_mean(values), 4) if values else None,
    }


# =============================================================================
# ETAP 2 — wymaga sentence-transformers
# =============================================================================

def _load_embeddings(texts: list):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        emb = model.encode(texts, normalize_embeddings=True)
        return emb, np
    except ImportError:
        return None, None


def semantic_diversity(debate_log: list) -> dict:
    """#8 Semantic Diversity — 1 − mean(cosine_similarity) między embeddingami."""
    texts = [e["text"] for e in debate_log]
    emb, np = _load_embeddings(texts)

    if emb is None:
        return {"overall": None, "per_round": {}, "note": "brak sentence-transformers"}

    n = len(emb)
    sims = [float(np.dot(emb[i], emb[j])) for i in range(n) for j in range(i + 1, n)]
    overall = round(1.0 - _mean(sims), 4) if sims else None

    by_round = {}
    for idx, e in enumerate(debate_log):
        by_round.setdefault(e["round"], []).append(idx)

    per_round = {}
    for rnd, idxs in by_round.items():
        if len(idxs) < 2:
            per_round[rnd] = None
            continue
        round_sims = [
            float(np.dot(emb[i], emb[j]))
            for ii, i in enumerate(idxs)
            for j in idxs[ii + 1:]
        ]
        per_round[rnd] = round(1.0 - _mean(round_sims), 4) if round_sims else None

    return {"overall": overall, "per_round": per_round}


def redundancy_ratio(debate_log: list, threshold: float = 0.85) -> dict:
    """#15 Redundancy Ratio — frakcja wypowiedzi bardzo podobna do wcześniejszych rund."""
    texts = [e["text"] for e in debate_log]
    emb, np = _load_embeddings(texts)

    if emb is None:
        return {"mean": None, "per_round": {}, "note": "brak sentence-transformers"}

    by_round = {}
    for idx, e in enumerate(debate_log):
        by_round.setdefault(e["round"], []).append(idx)

    history_idxs = []
    per_round = {}

    for rnd in sorted(by_round.keys()):
        curr_idxs = by_round[rnd]
        if not history_idxs:
            per_round[rnd] = 0.0
        else:
            redundant = sum(
                1 for ci in curr_idxs
                if max(float(np.dot(emb[ci], emb[hi])) for hi in history_idxs) >= threshold
            )
            per_round[rnd] = round(redundant / len(curr_idxs), 4)
        history_idxs.extend(curr_idxs)

    values = [v for v in per_round.values() if v is not None]
    return {
        "mean": round(_mean(values), 4) if values else None,
        "per_round": per_round,
        "threshold": threshold,
    }


# =============================================================================
# Funkcja zbiorcza — wywołana przez main.py
# =============================================================================

def compute_all(debate_log: list, decision_result: dict, config: dict) -> dict:
    """Liczy wszystkie dostępne metryki i zwraca jeden słownik."""
    return {
        "tokens_per_turn": tokens_per_turn(debate_log),
        "flip_rate": flip_rate(debate_log),
        "entropy": entropy(debate_log),
        "convergence": convergence_round(
            decision_result,
            debate_log,
            threshold=config.get("consensus_threshold", 0.5),
        ),
        "auc_agreement": auc_agreement(debate_log),
        "semantic_diversity": semantic_diversity(debate_log),
        "redundancy_ratio": redundancy_ratio(debate_log),
    }


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