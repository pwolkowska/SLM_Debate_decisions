"""
metrics.py — Obliczanie metryk z ustrukturyzowanych danych debaty.

Metryki podstawowe (dawny metrics.py):
    tokens_per_turn                      — czy osobowość wpływa na ilość produkowanego tekstu?
    opinion_shift                        — semantyczna zmiana stanowiska (cosine distance)
    opinion_shift_sentiment              — zmiana sentymentu runda po rundzie per agent
    between_agent_similarity             — czy agenci produkują różne argumenty i czy się zbliżają?
    between_agent_similarity_sentiment   — porównanie sentymentu między agentami per runda
    lexical_richness                     — czy osobowość wpływa na różnorodność słownictwa?
    convergence                          — czy debata kończy się porozumieniem i jak szybko?
    answer_flip_rate                     — NoF i ToF (sycophancy / position stability)
    history_repetition                   — czy agent zaczyna od powtarzania historii debaty?
    semantic_diversity                   — globalna różnorodność semantyczna wypowiedzi
    redundancy_ratio                     — frakcja wypowiedzi redundantnych relative do wcześniejszych rund

Metryki rozszerzone (dawny metrics_j.py):
    argument_novelty               — nowość argumentu agenta względem jego własnej historii
    claim_grounding                — ugruntowanie twierdzeń w faktach/danych (soft-match lub regex)
    question_density               — zagęszczenie pytań (strategie sokratejskie vs brak stanowiska)
    direct_address_rate            — jak często agent odnosi się bezpośrednio do innego agenta
    rebuttal_depth                 — czy agent odpowiada na konkretny argument poprzednika
    reciprocal_shift_index         — korelacja zmian stanowiska między agentami
    opinion_trajectory_monotonicity— czy zmiany stanowiska są monotoniczne czy oscylacyjne
    total_opinion_drift            — całkowite oddalenie od pozycji startowej per agent
    convergence_speed              — ile rund do semantycznego zbliżenia (metryka relatywna)
    hedging_rate                   — częstotliwość markerów niepewności (soft-match lub regex)
    assertiveness_score            — częstotliwość markerów pewności (soft-match lub regex)
    turn_length_entropy            — entropia rozkładu długości tokenów per agent
    gini_speaking_time             — nierówność czasu mówienia (Gini coefficient)

Zależności opcjonalne:
    sentence-transformers  →  opinion_shift, between_agent_similarity, answer_flip_rate,
                               argument_novelty, claim_grounding, rebuttal_depth,
                               reciprocal_shift_index, convergence_speed,
                               total_opinion_drift,
                               hedging_rate (tryb soft), assertiveness_score (tryb soft)
    transformers           →  opinion_shift_sentiment, between_agent_similarity_sentiment

Źródła:
    Hong et al. 2025, SYCON Bench (arXiv:2505.23840)
    Laban et al. 2023, FlipFlop (arXiv:2311.08596)
"""

import math
import re
from collections import defaultdict
from typing import Optional


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
    """Zwraca (embeddings_array, numpy_module) lub (None, None) jeśli brak biblioteki."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        emb = model.encode(texts, normalize_embeddings=True)
        return emb, np
    except Exception:
        return None, None

def _load_sentiment_pipeline():
    """
    Ładuje pipeline analizy sentymentu (wielojęzyczny).
    Zwraca pipeline lub None jeśli brak transformers.

    Domyślny model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
    (obsługuje polski; 3 klasy: positive / neutral / negative)
    """
    try:
        from transformers import pipeline
        sentiment = pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        return sentiment
    except Exception:
        return None

def _sentiment_score(pipeline_result) -> Optional[float]:
    """
    Zamienia wynik pipeline (lista dict z 'label'/'score') na skalar [-1, +1].

    Konwencja:
        positive → +score
        neutral  →  0
        negative → −score
    Zwraca None jeśli wynik jest pusty lub nieznany.
    """
    if not pipeline_result:
        return None
    label_map = {}
    for item in pipeline_result:
        label_map[item["label"].lower()] = item["score"]

    pos = label_map.get("positive", 0.0)
    neg = label_map.get("negative", 0.0)
    return round(pos - neg, 4)

def _ngrams(text: str, n: int) -> set:
    """Zwraca zbiór n-gramów słownych z tekstu."""
    words = re.findall(r"\b\w+\b", text.lower())
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()

def _ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Jaccard overlap n-gramów między dwoma tekstami. 0..1."""
    a, b = _ngrams(text_a, n), _ngrams(text_b, n)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)

def _prefix(text: str, words: int = 60) -> str:
    """Zwraca pierwsze `words` słów tekstu."""
    return " ".join(text.split()[:words])

def _cosine_sim(emb_a, emb_b, np) -> float:
    return float(np.dot(emb_a, emb_b))

def _sentences(text: str) -> list[str]:
    """Dzieli tekst na zdania po kropce/wykrzykniku/pytajniku."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# =============================================================================
# Listy kotwic (anchors) dla soft-matchingu
# =============================================================================

GROUNDING_ANCHORS = [
    "badania pokazują", "dane wskazują", "według badań", "zgodnie z danymi",
    "badania naukowe", "statystyki pokazują", "raport wskazuje",
    "udowodniono naukowo", "wyniki badań", "evidence suggests",
    "studies show", "research indicates", "according to data",
    "statistics show", "it has been proven", "the data shows",
    "na podstawie danych", "w świetle faktów", "fakty wskazują",
]

HEDGING_ANCHORS = [
    "być może", "wydaje się", "chyba", "prawdopodobnie", "nie jestem pewien",
    "możliwe że", "zdaje się", "mogłoby się wydawać", "nie wiem czy",
    "trudno powiedzieć", "maybe", "perhaps", "it seems", "possibly",
    "i'm not sure", "might be", "could be", "it appears", "seemingly",
    "niewykluczone", "w pewnym sensie", "można by sądzić",
]

ASSERTIVENESS_ANCHORS = [
    "zdecydowanie", "na pewno", "jestem przekonany", "faktem jest",
    "bez wątpienia", "absolutnie", "jestem pewien", "nie ulega wątpliwości",
    "oczywiście", "definitely", "certainly", "i am convinced", "it is a fact",
    "without doubt", "absolutely", "undoubtedly", "clearly", "obviously",
    "bezspornie", "niezbicie", "z całą pewnością",
]


def _anchor_centroid(anchors: list[str], np):
    """Zwraca centroid embeddingów listy anchor-fraz lub None."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embs = model.encode(anchors, normalize_embeddings=True)
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid
    except Exception:
        return None

def _soft_match_sentences(text: str, centroid, np, threshold: float = 0.45) -> tuple[int, int]:
    """
    Dla każdego zdania w tekście liczy cosine similarity do centroidu.
    Zwraca (matched_count, total_sentences).
    """
    if centroid is None or np is None:
        return 0, 0
    sents = _sentences(text)
    if not sents:
        return 0, 0
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embs = model.encode(sents, normalize_embeddings=True)
        sims = [float(np.dot(embs[i], centroid)) for i in range(len(sents))]
        matched = sum(1 for s in sims if s >= threshold)
        return matched, len(sents)
    except Exception:
        return 0, len(sents)

def _regex_match_sentences(text: str, patterns: list[str]) -> tuple[int, int]:
    """Zlicza zdania zawierające przynajmniej jeden wzorzec z listy."""
    sents = _sentences(text)
    if not sents:
        return 0, 0
    combined = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    matched = sum(1 for s in sents if combined.search(s))
    return matched, len(sents)


# =============================================================================
# METRYKI PODSTAWOWE
# =============================================================================

def history_repetition(
    debate_log: list,
    *,
    ngram_threshold: float = 0.25,
    embed_threshold: float = 0.85,
    prefix_words: int = 60,
    ngram_n: int = 3,
    embeddings=None,
    np=None,
) -> dict:
    """
    Sprawdza, czy agent zaczyna wypowiedź od powtórzenia historii debaty.

    Dla każdej wypowiedzi (poza pierwszą rundą) porównuje prefix agenta
    z wszystkimi poprzednimi wypowiedziami:
      - ngram_overlap  ≥ ngram_threshold  → powtórzenie dosłowne
      - cosine_sim     ≥ embed_threshold  → powtórzenie semantyczne (wymaga embeddings)

    Zwraca
    -------
    dict z kluczami:
        per_entry : lista dict {round, agent, ngram_hit, embed_hit, max_ngram, max_cosine}
        per_agent : {agent: {repetition_count, total_turns, repetition_rate}}
        summary   : {total_repetitions, total_turns, overall_rate}
    """
    per_entry = []
    history_texts: list[str] = []
    history_indices: list[int] = []

    for idx, entry in enumerate(debate_log):
        current_prefix = _prefix(entry["text"], prefix_words)

        if not history_texts:
            per_entry.append({
                "round": entry["round"],
                "agent": entry["agent"],
                "ngram_hit": False,
                "embed_hit": False,
                "max_ngram": None,
                "max_cosine": None,
            })
            history_texts.append(entry["text"])
            history_indices.append(idx)
            continue

        ngram_scores = [
            _ngram_overlap(current_prefix, hist, ngram_n)
            for hist in history_texts
        ]
        max_ngram = round(max(ngram_scores), 4)
        ngram_hit = max_ngram >= ngram_threshold

        max_cosine = None
        embed_hit = False
        if embeddings is not None and np is not None:
            current_emb = embeddings[idx]
            cos_scores = [
                float(np.dot(current_emb, embeddings[h_idx]))
                for h_idx in history_indices
            ]
            max_cosine = round(max(cos_scores), 4)
            embed_hit = max_cosine >= embed_threshold

        per_entry.append({
            "round": entry["round"],
            "agent": entry["agent"],
            "ngram_hit": ngram_hit,
            "embed_hit": embed_hit,
            "max_ngram": max_ngram,
            "max_cosine": max_cosine,
        })
        history_texts.append(entry["text"])
        history_indices.append(idx)

    by_agent: dict[str, dict] = defaultdict(lambda: {"repetition_count": 0, "total_turns": 0})
    for e in per_entry:
        ag = e["agent"]
        by_agent[ag]["total_turns"] += 1
        if e["ngram_hit"] or e["embed_hit"]:
            by_agent[ag]["repetition_count"] += 1

    per_agent = {}
    for agent, stats in by_agent.items():
        total = stats["total_turns"]
        count = stats["repetition_count"]
        per_agent[agent] = {
            "repetition_count": count,
            "total_turns": total,
            "repetition_rate": round(count / total, 4) if total else None,
        }

    total_turns = len(per_entry)
    total_reps = sum(1 for e in per_entry if e["ngram_hit"] or e["embed_hit"])

    return {
        "per_entry": per_entry,
        "per_agent": per_agent,
        "summary": {
            "total_repetitions": total_reps,
            "total_turns": total_turns,
            "overall_rate": round(total_reps / total_turns, 4) if total_turns else None,
            "ngram_threshold": ngram_threshold,
            "embed_threshold": embed_threshold,
            "prefix_words": prefix_words,
        },
    }


def opinion_shift_sentiment(debate_log: list, sentiment_pipeline) -> dict:
    """
    Analiza zmiany sentymentu agenta runda po rundzie.

    Zwraca
    -------
        per_agent : {agent: {trajectory, delta_first_last, mean_shift_abs}}
    """
    if sentiment_pipeline is None:
        return {
            "per_agent": {},
            "note": "brak transformers lub nie udało się załadować modelu sentymentu",
        }

    by_agent: dict[str, list] = defaultdict(list)
    for entry in debate_log:
        by_agent[entry["agent"]].append(entry)

    per_agent = {}
    for agent, entries in by_agent.items():
        texts = [e["text"] for e in entries]
        rounds = [e["round"] for e in entries]

        try:
            raw_scores = sentiment_pipeline(texts)
        except Exception as exc:
            per_agent[agent] = {"error": str(exc)}
            continue

        scores = [_sentiment_score(r) for r in raw_scores]
        trajectory = [
            {"round": rnd, "sentiment": sc}
            for rnd, sc in zip(rounds, scores)
        ]

        valid = [s for s in scores if s is not None]
        if len(valid) >= 2:
            delta = round(valid[-1] - valid[0], 4)
            abs_shifts = [round(abs(valid[i+1] - valid[i]), 4) for i in range(len(valid) - 1)]
            mean_abs = _mean(abs_shifts)
        else:
            delta = None
            mean_abs = None

        per_agent[agent] = {
            "trajectory": trajectory,
            "delta_first_last": delta,
            "mean_shift_abs": mean_abs,
        }

    return {"per_agent": per_agent}


def between_agent_similarity_sentiment(
    debate_log: list,
    sentiment_pipeline,
    history_rep_result: Optional[dict] = None,
) -> dict:
    """
    Porównuje sentyment między agentami w każdej rundzie.

    Zwraca
    -------
        per_round    : {runda: {agent_sentiments, mean_pair_diff, agents_with_repetition}}
        initial      : mean_pair_diff rundy 1
        final        : mean_pair_diff ostatniej rundy
        delta        : final − initial
    """
    if sentiment_pipeline is None:
        return {
            "per_round": {},
            "initial": None,
            "final": None,
            "delta": None,
            "note": "brak transformers lub nie udało się załadować modelu sentymentu",
        }

    rep_map: dict[tuple, bool] = {}
    if history_rep_result:
        for e in history_rep_result.get("per_entry", []):
            rep_map[(e["round"], e["agent"])] = e["ngram_hit"] or e["embed_hit"]

    by_round: dict[int, list] = defaultdict(list)
    for entry in debate_log:
        by_round[entry["round"]].append(entry)

    per_round = {}
    for rnd, entries in sorted(by_round.items()):
        texts = [e["text"] for e in entries]
        agents = [e["agent"] for e in entries]

        try:
            raw = sentiment_pipeline(texts)
        except Exception as exc:
            per_round[rnd] = {"error": str(exc)}
            continue

        scores = [_sentiment_score(r) for r in raw]
        agent_scores = {ag: sc for ag, sc in zip(agents, scores)}

        agent_list = list(agent_scores.keys())
        diffs = []
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                s_i = agent_scores[agent_list[i]]
                s_j = agent_scores[agent_list[j]]
                if s_i is not None and s_j is not None:
                    diffs.append(abs(s_i - s_j))

        reps_in_round = [ag for ag in agents if rep_map.get((rnd, ag), False)]

        per_round[rnd] = {
            "agent_sentiments": {
                ag: {
                    "sentiment": sc,
                    "is_repetition": rep_map.get((rnd, ag), False),
                }
                for ag, sc in agent_scores.items()
            },
            "mean_pair_diff": round(_mean(diffs), 4) if diffs else None,
            "agents_with_repetition": reps_in_round,
        }

    valid = {r: v["mean_pair_diff"] for r, v in per_round.items() if isinstance(v.get("mean_pair_diff"), float)}
    if valid:
        sorted_rnds = sorted(valid.keys())
        initial = valid[sorted_rnds[0]]
        final = valid[sorted_rnds[-1]]
        delta = round(final - initial, 4)
    else:
        initial = final = delta = None

    return {
        "per_round": per_round,
        "initial": initial,
        "final": final,
        "delta": delta,
    }


def answer_flip_rate(
    debate_log: list,
    embeddings,
    np,
    *,
    flip_threshold: float = 0.20,
) -> dict:
    """
    NoF (Number of Flips) i ToF (Turn of First Flip) per agent.

    Definicje (Hong et al. 2025; Laban et al. 2023):
        NoF_agent = |{ t : position_t ≠ position_{t-1} }|
        ToF_agent = min{ t : position_t ≠ position_1 }

    Wykrywanie zmiany pozycji przez cosine distance między kolejnymi wypowiedziami.

    Zwraca
    -------
        per_agent : {agent: {nof, tof, flip_rounds, trajectory}}
        summary   : {mean_nof, max_nof, agents_with_flip, flip_threshold}
    """
    if embeddings is None or np is None:
        return {
            "per_agent": {},
            "summary": {},
            "note": "brak sentence-transformers; NoF/ToF niedostępne",
        }

    by_agent: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        by_agent[entry["agent"]].append((idx, entry["round"]))

    per_agent = {}
    for agent, entries in by_agent.items():
        if len(entries) < 2:
            per_agent[agent] = {
                "nof": 0,
                "tof": None,
                "flip_rounds": [],
                "trajectory": [{"round": r, "cos_dist_prev": None} for _, r in entries],
                "note": "za mało wypowiedzi (< 2)",
            }
            continue

        trajectory = []
        flip_rounds = []
        nof = 0
        tof = None

        prev_idx, prev_round = entries[0]
        trajectory.append({"round": prev_round, "cos_dist_prev": None, "flip": False})

        for turn_num, (cur_idx, cur_round) in enumerate(entries[1:], start=1):
            cos_dist = round(1.0 - _cosine_sim(embeddings[cur_idx], embeddings[prev_idx], np), 4)
            is_flip = cos_dist >= flip_threshold

            if is_flip:
                nof += 1
                flip_rounds.append(cur_round)
                if tof is None:
                    tof = cur_round

            trajectory.append({
                "round": cur_round,
                "cos_dist_prev": cos_dist,
                "flip": is_flip,
            })
            prev_idx = cur_idx

        per_agent[agent] = {
            "nof": nof,
            "tof": tof,
            "flip_rounds": flip_rounds,
            "trajectory": trajectory,
        }

    nof_values = [v["nof"] for v in per_agent.values() if isinstance(v.get("nof"), int)]
    agents_with_flip = sum(1 for v in per_agent.values() if isinstance(v.get("nof"), int) and v["nof"] > 0)

    return {
        "per_agent": per_agent,
        "summary": {
            "mean_nof": _mean(nof_values),
            "max_nof": max(nof_values) if nof_values else None,
            "agents_with_flip": agents_with_flip,
            "flip_threshold": flip_threshold,
        },
    }


def tokens_per_turn(debate_log: list) -> dict:
    """Verbosity per agent i overall."""
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
    """Semantyczna zmiana stanowiska: 1 − cosine_sim(pierwsza, ostatnia) per agent."""
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
        cosine_sim = float(np.dot(embeddings[idxs[0]], embeddings[idxs[-1]]))
        per_agent[agent] = round(1.0 - cosine_sim, 4)

    valid = [v for v in per_agent.values() if v is not None]
    return {"per_agent": per_agent, "mean": _mean(valid)}


def between_agent_similarity(debate_log: list, embeddings, np) -> dict:
    """Podobieństwo semantyczne między agentami per runda."""
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
    """Type-Token Ratio per agent."""
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
    """Wynik protokołu consensus."""
    return {
        "reached": decision_result.get("consensus_reached", False),
        "convergence_round": decision_result.get("convergence_round", None),
        "agreement_ratios": [
            r["agreement_ratio"]
            for r in decision_result.get("consensus_rounds", [])
        ],
    }


def semantic_diversity(debate_log: list, embeddings, np) -> dict:
    """
    1 − mean(pairwise cos_sim) dla wszystkich par wypowiedzi debaty.
    Źródło: Liang et al. 2023 (arXiv:2305.19118) — metryka #8
    """
    if embeddings is None:
        return {"overall": None, "per_round": {}, "note": "brak sentence-transformers"}

    n = len(embeddings)
    if n < 2:
        return {"overall": None, "per_round": {}}

    all_pairs = [
        float(np.dot(embeddings[i], embeddings[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    overall = round(1.0 - _mean(all_pairs), 4)

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
        per_round[rnd] = round(1.0 - _mean(pairs), 4) if pairs else None

    return {"overall": overall, "per_round": per_round}


def redundancy_ratio(debate_log: list, embeddings, np, threshold: float = 0.85) -> dict:
    """
    Frakcja wypowiedzi w rundzie t semantycznie podobnych (cos_sim > threshold)
    do jakiejkolwiek wypowiedzi z wcześniejszych rund.
    Źródło: Liang et al. 2023 (arXiv:2305.19118) — metryka #15
    """
    if embeddings is None:
        return {"per_round": {}, "mean": None, "note": "brak sentence-transformers"}

    by_round = defaultdict(list)
    for idx, e in enumerate(debate_log):
        by_round[e["round"]].append(idx)

    sorted_rounds = sorted(by_round.keys())
    per_round = {}

    for i, rnd in enumerate(sorted_rounds):
        if i == 0:
            per_round[rnd] = 0.0
            continue
        prior = [idx for r in sorted_rounds[:i] for idx in by_round[r]]
        curr  = by_round[rnd]
        redundant = sum(
            1 for c in curr
            if max(float(np.dot(embeddings[c], embeddings[p])) for p in prior) > threshold
        )
        per_round[rnd] = round(redundant / len(curr), 4) if curr else 0.0

    return {"per_round": per_round, "mean": _mean(list(per_round.values())), "threshold": threshold}


# =============================================================================
# METRYKI ROZSZERZONE (dawny metrics_j.py)
# =============================================================================

def argument_novelty(debate_log: list, embeddings, np) -> dict:
    """
    Nowość argumentu agenta względem jego własnej historii.

    Dla wypowiedzi t agenta A: cosine_distance(embedding_t, centroid(embedding_1..t-1)).

    Zwraca
    -------
        per_agent : {agent: {per_turn: [{round, novelty}], mean, std}}
        overall   : mean novelty across all agents and turns
    """
    if embeddings is None or np is None:
        return {"per_agent": {}, "overall": None,
                "note": "brak sentence-transformers; argument_novelty niedostępne"}

    by_agent: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        by_agent[entry["agent"]].append((idx, entry["round"]))

    per_agent = {}
    all_novelties = []

    for agent, entries in by_agent.items():
        turns = []
        for i, (cur_idx, cur_round) in enumerate(entries):
            if i == 0:
                turns.append({"round": cur_round, "novelty": None})
                continue
            prev_embs = np.stack([embeddings[entries[j][0]] for j in range(i)])
            centroid = prev_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            cos_sim = float(np.dot(embeddings[cur_idx], centroid))
            novelty = round(1.0 - cos_sim, 4)
            turns.append({"round": cur_round, "novelty": novelty})
            all_novelties.append(novelty)

        valid = [t["novelty"] for t in turns if t["novelty"] is not None]
        per_agent[agent] = {
            "per_turn": turns,
            "mean": _mean(valid),
            "std": _std(valid),
        }

    return {"per_agent": per_agent, "overall": _mean(all_novelties)}


def claim_grounding(
    debate_log: list,
    np=None,
    *,
    mode: str = "auto",
    threshold: float = 0.45,
) -> dict:
    """
    Frakcja zdań zawierających markery evidencyjności (fakty, dane, badania).

    Tryby:
        "soft"  — cosine similarity do centroidu GROUNDING_ANCHORS (wymaga ST)
        "regex" — dopasowanie do listy wzorców
        "auto"  — soft jeśli sentence-transformers dostępne, inaczej regex

    Zwraca
    -------
        per_agent : {agent: {grounded_sentences, total_sentences, grounding_rate}}
        overall   : mean grounding_rate
        mode_used : "soft" lub "regex"
    """
    use_soft = False
    if mode in ("soft", "auto") and np is not None:
        centroid = _anchor_centroid(GROUNDING_ANCHORS, np)
        use_soft = centroid is not None
    else:
        centroid = None

    by_agent: dict[str, list] = defaultdict(list)
    for entry in debate_log:
        by_agent[entry["agent"]].append(entry["text"])

    per_agent = {}
    all_rates = []

    for agent, texts in by_agent.items():
        combined = " ".join(texts)
        if use_soft:
            matched, total = _soft_match_sentences(combined, centroid, np, threshold)
        else:
            matched, total = _regex_match_sentences(combined, GROUNDING_ANCHORS)

        rate = round(matched / total, 4) if total else None
        per_agent[agent] = {
            "grounded_sentences": matched,
            "total_sentences": total,
            "grounding_rate": rate,
        }
        if rate is not None:
            all_rates.append(rate)

    return {
        "per_agent": per_agent,
        "overall": _mean(all_rates),
        "mode_used": "soft" if use_soft else "regex",
        "threshold": threshold if use_soft else None,
    }


def question_density(debate_log: list) -> dict:
    """
    Frakcja zdań będących pytaniami per agent i per runda.

    Zwraca
    -------
        per_agent : {agent: {question_count, total_sentences, question_rate}}
        per_round : {round: mean question_rate across agents}
        overall   : mean across all turns
    """
    by_agent: dict[str, dict] = defaultdict(lambda: {"q": 0, "total": 0})
    by_round: dict[int, list] = defaultdict(list)
    all_rates = []

    for entry in debate_log:
        text = entry["text"]
        sents = _sentences(text)
        questions = sum(1 for s in re.split(r"[.!?]+", text) if "?" in s and s.strip())
        total = max(len(sents), 1)
        rate = round(questions / total, 4)

        by_agent[entry["agent"]]["q"] += questions
        by_agent[entry["agent"]]["total"] += total
        by_round[entry["round"]].append(rate)
        all_rates.append(rate)

    per_agent = {}
    for agent, d in by_agent.items():
        t = d["total"]
        q = d["q"]
        per_agent[agent] = {
            "question_count": q,
            "total_sentences": t,
            "question_rate": round(q / t, 4) if t else None,
        }

    per_round = {r: _mean(rates) for r, rates in by_round.items()}

    return {
        "per_agent": per_agent,
        "per_round": per_round,
        "overall": _mean(all_rates),
    }


def direct_address_rate(debate_log: list) -> dict:
    """
    Jak często agent bezpośrednio adresuje innego agenta lub odnosi się
    do jego wypowiedzi przez markery interakcji.

    Zwraca
    -------
        per_agent         : {agent: {direct_turns, interaction_marker_turns,
                                      total_turns, address_rate,
                                      interaction_rate, combined_rate,
                                      addressed_agents: {agent: count}}}
        most_addressed    : agent najczęściej adresowany z imienia
        mean_address_rate : średnia address_rate (tylko imienne)
        mean_combined_rate: średnia combined_rate (imienne + markery)
    """
    INTERACTION_MARKERS = [
        "zgadzam się", "nie zgadzam się", "zgadzam sie", "nie zgadzam sie",
        "jak powiedziałeś", "jak powiedziałes", "twój argument", "twoj argument",
        "masz rację", "masz racje", "jednak ", "to prawda", "nie sądzę",
        "nie sadze", "wręcz przeciwnie", "wrécz przeciwnie", "wrecz przeciwnie",
        "w odpowiedzi na to", "odnosząc się do tego", "odnosac sie do tego",
        "twoje stanowisko", "twoja propozycja", "twoje zdanie",
        "you're right", "i agree", "i disagree", "your argument",
        "as you said", "however,", "on the contrary", "in response to that",
    ]
    interaction_pattern = re.compile(
        "|".join(re.escape(m) for m in INTERACTION_MARKERS), re.IGNORECASE
    )

    agent_names = list({e["agent"] for e in debate_log})

    by_agent: dict[str, dict] = defaultdict(lambda: {
        "direct_turns": 0,
        "interaction_marker_turns": 0,
        "total_turns": 0,
        "addressed_agents": defaultdict(int),
    })

    for entry in debate_log:
        agent = entry["agent"]
        text = entry["text"].lower()
        by_agent[agent]["total_turns"] += 1
        addressed_by_name = False
        has_interaction_marker = False

        for other in agent_names:
            if other == agent:
                continue
            other_lower = other.lower()
            if other_lower in text:
                by_agent[agent]["addressed_agents"][other] += 1
                addressed_by_name = True
            cite_patterns = [
                rf"jak {re.escape(other_lower)}",
                rf"{re.escape(other_lower)} stwierdzi",
                rf"{re.escape(other_lower)} twierdzi",
                rf"{re.escape(other_lower)} powiedzia",
                rf"nawiązując do {re.escape(other_lower)}",
                rf"w odpowiedzi na {re.escape(other_lower)}",
                rf"as {re.escape(other_lower)}",
                rf"{re.escape(other_lower)} said",
                rf"{re.escape(other_lower)} argued",
                rf"according to {re.escape(other_lower)}",
            ]
            if any(re.search(p, text) for p in cite_patterns):
                by_agent[agent]["addressed_agents"][other] += 1
                addressed_by_name = True

        if addressed_by_name:
            by_agent[agent]["direct_turns"] += 1

        if interaction_pattern.search(text):
            has_interaction_marker = True
        if has_interaction_marker:
            by_agent[agent]["interaction_marker_turns"] += 1

    per_agent = {}
    all_address_rates = []
    all_combined_rates = []
    address_received: dict[str, int] = defaultdict(int)

    for agent, d in by_agent.items():
        t = d["total_turns"]
        dt = d["direct_turns"]
        im = d["interaction_marker_turns"]
        combined_turns = sum(
            1 for entry in debate_log
            if entry["agent"] == agent and (
                any(
                    other.lower() in entry["text"].lower()
                    for other in agent_names if other != agent
                ) or interaction_pattern.search(entry["text"].lower())
            )
        )
        address_rate = round(dt / t, 4) if t else None
        interaction_rate = round(im / t, 4) if t else None
        combined_rate = round(combined_turns / t, 4) if t else None

        per_agent[agent] = {
            "direct_turns": dt,
            "interaction_marker_turns": im,
            "total_turns": t,
            "address_rate": address_rate,
            "interaction_rate": interaction_rate,
            "combined_rate": combined_rate,
            "addressed_agents": dict(d["addressed_agents"]),
        }
        if address_rate is not None:
            all_address_rates.append(address_rate)
        if combined_rate is not None:
            all_combined_rates.append(combined_rate)
        for target, count in d["addressed_agents"].items():
            address_received[target] += count

    most_addressed = max(address_received, key=address_received.get) if address_received else None

    return {
        "per_agent": per_agent,
        "most_addressed": most_addressed,
        "address_received_counts": dict(address_received),
        "mean_address_rate": _mean(all_address_rates),
        "mean_combined_rate": _mean(all_combined_rates),
    }


def rebuttal_depth(debate_log: list, embeddings, np) -> dict:
    """
    Czy agent odpowiada na konkretny argument bezpośredniego poprzednika.

    Dla każdej wypowiedzi (poza pierwszą) liczy cosine_sim(t, t-1).

    Zwraca
    -------
        per_entry : [{round, agent, rebuttal_sim, prev_agent}]
        per_agent : {agent: {mean_rebuttal_sim, std}}
        overall   : mean across all entries
    """
    if embeddings is None or np is None:
        return {"per_entry": [], "per_agent": {}, "overall": None,
                "note": "brak sentence-transformers; rebuttal_depth niedostępne"}

    per_entry = []
    by_agent: dict[str, list] = defaultdict(list)

    for idx, entry in enumerate(debate_log):
        if idx == 0:
            per_entry.append({
                "round": entry["round"],
                "agent": entry["agent"],
                "rebuttal_sim": None,
                "prev_agent": None,
            })
            continue
        prev = debate_log[idx - 1]
        sim = round(float(np.dot(embeddings[idx], embeddings[idx - 1])), 4)
        per_entry.append({
            "round": entry["round"],
            "agent": entry["agent"],
            "rebuttal_sim": sim,
            "prev_agent": prev["agent"],
        })
        by_agent[entry["agent"]].append(sim)

    per_agent = {}
    all_sims = []
    for agent, sims in by_agent.items():
        per_agent[agent] = {"mean_rebuttal_sim": _mean(sims), "std": _std(sims)}
        all_sims.extend(sims)

    return {
        "per_entry": per_entry,
        "per_agent": per_agent,
        "overall": _mean(all_sims),
    }


def reciprocal_shift_index(debate_log: list, embeddings, np) -> dict:
    """
    Korelacja zmian stanowiska między agentami round-by-round (Pearson r).

    r ≈ +1 → agenci zmieniają się synchronicznie (reaktywna debata)
    r ≈ -1 → zmiany antyfazowe
    r ≈  0 → zmiany niezależne (monologiczna debata)

    Zwraca
    -------
        per_pair   : {(A, B): {pearson_r, n_rounds}}
        mean_abs_r : średnia |r| (miara reaktywności debaty)
    """
    if embeddings is None or np is None:
        return {"per_pair": {}, "mean_r": None,
                "note": "brak sentence-transformers; reciprocal_shift_index niedostępne"}

    by_agent: dict[str, dict] = defaultdict(dict)
    agent_turns: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        agent_turns[entry["agent"]].append((idx, entry["round"]))

    for agent, entries in agent_turns.items():
        for i, (cur_idx, cur_round) in enumerate(entries):
            if i == 0:
                continue
            prev_idx = entries[i - 1][0]
            cos_dist = round(1.0 - float(np.dot(embeddings[cur_idx], embeddings[prev_idx])), 4)
            by_agent[agent][cur_round] = cos_dist

    agents = list(by_agent.keys())
    per_pair = {}

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            common_rounds = sorted(set(by_agent[a].keys()) & set(by_agent[b].keys()))
            n = len(common_rounds)
            if n < 2:
                per_pair[f"{a}|{b}"] = {
                    "pearson_r": None, "n_rounds": n,
                    "note": f"za mało wspólnych rund ({n} < 2)"
                }
                continue

            xa = [by_agent[a][r] for r in common_rounds]
            xb = [by_agent[b][r] for r in common_rounds]

            ma, mb = sum(xa) / n, sum(xb) / n
            num = sum((xa[k] - ma) * (xb[k] - mb) for k in range(n))
            da = (sum((x - ma) ** 2 for x in xa)) ** 0.5
            db = (sum((x - mb) ** 2 for x in xb)) ** 0.5
            if da == 0 or db == 0:
                r = 0.0
            else:
                r = round(num / (da * db), 4)

            per_pair[f"{a}|{b}"] = {"pearson_r": r, "n_rounds": n}
            if n == 2:
                per_pair[f"{a}|{b}"]["note"] = "tylko 2 rundy — r zawsze ±1, niska wiarygodność"

    valid_r = [abs(v["pearson_r"]) for v in per_pair.values()
               if v.get("pearson_r") is not None]

    return {
        "per_pair": per_pair,
        "mean_abs_r": _mean(valid_r),
    }


def opinion_trajectory_monotonicity(debate_log: list, embeddings, np) -> dict:
    """
    Czy zmiany stanowiska agenta są monotoniczne (dryfowanie) czy oscylacyjne.

    Wskaźnik M ∈ [-1, +1]:
        M = (rośnie - maleje) / (rośnie + maleje)
        M = +1 → ciągłe oddalanie, M = -1 → powrót, M ≈ 0 → oscylacje

    Bez embeddings — używa długości tokenów jako proxy.

    Zwraca
    -------
        per_agent   : {agent: {monotonicity, direction_changes, n_turns, trajectory}}
        overall_abs : mean |monotonicity|
    """
    if embeddings is None or np is None:
        by_agent: dict[str, list] = defaultdict(list)
        for entry in debate_log:
            by_agent[entry["agent"]].append((entry["round"], entry["tokens"]))

        per_agent = {}
        all_mono = []
        for agent, turns in by_agent.items():
            diffs = [turns[i+1][1] - turns[i][1] for i in range(len(turns) - 1)]
            if not diffs:
                per_agent[agent] = {"monotonicity": None, "n_turns": len(turns),
                                     "note": "za mało tur", "proxy": "verbosity"}
                continue
            inc = sum(1 for d in diffs if d > 0)
            dec = sum(1 for d in diffs if d < 0)
            total = inc + dec
            mono = round((inc - dec) / total, 4) if total else 0.0
            per_agent[agent] = {
                "monotonicity": mono,
                "direction_changes": sum(
                    1 for i in range(len(diffs)-1) if (diffs[i] > 0) != (diffs[i+1] > 0)
                ),
                "n_turns": len(turns),
                "proxy": "verbosity",
            }
            all_mono.append(abs(mono))

        return {"per_agent": per_agent, "overall_abs": _mean(all_mono),
                "note": "brak embeddings, użyto proxy verbosity"}

    agent_turns: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        agent_turns[entry["agent"]].append((idx, entry["round"]))

    per_agent = {}
    all_mono = []

    for agent, entries in agent_turns.items():
        if len(entries) < 2:
            per_agent[agent] = {"monotonicity": None, "n_turns": len(entries),
                                  "note": "za mało tur"}
            continue

        traj = []
        dists = []
        for i, (cur_idx, cur_round) in enumerate(entries):
            if i == 0:
                traj.append({"round": cur_round, "cos_dist": None})
                continue
            prev_idx = entries[i-1][0]
            d = round(1.0 - float(np.dot(embeddings[cur_idx], embeddings[prev_idx])), 4)
            traj.append({"round": cur_round, "cos_dist": d})
            dists.append(d)

        if not dists:
            per_agent[agent] = {"monotonicity": None, "trajectory": traj,
                                  "n_turns": len(entries), "note": "za mało tur"}
            continue

        diffs = [dists[i+1] - dists[i] for i in range(len(dists) - 1)]
        inc = sum(1 for d in diffs if d > 0)
        dec = sum(1 for d in diffs if d < 0)
        total = inc + dec
        mono = round((inc - dec) / total, 4) if total else 0.0

        dir_changes = sum(
            1 for i in range(len(diffs)-1)
            if (diffs[i] > 0) != (diffs[i+1] > 0) and diffs[i] != 0 and diffs[i+1] != 0
        )

        per_agent[agent] = {
            "monotonicity": mono,
            "direction_changes": dir_changes,
            "n_turns": len(entries),
            "trajectory": traj,
        }
        all_mono.append(abs(mono))

    return {"per_agent": per_agent, "overall_abs": _mean(all_mono)}


def total_opinion_drift(debate_log: list, embeddings, np) -> dict:
    """
    Całkowite oddalenie agenta od pozycji startowej: cosine_distance(R1, Rn).

    Uzupełnienie `opinion_shift` (lokalna delta) i `opinion_trajectory_monotonicity`
    (kierunek zmian). Ta metryka odpowiada na pytanie: *o ile dalej* od punktu
    startowego agent skończył debatę.

    Zwraca
    -------
        per_agent : {agent: {drift, first_round, last_round}}
        mean      : średni drift
        max_drift : {agent, value}
    """
    if embeddings is None or np is None:
        return {
            "per_agent": {},
            "mean": None,
            "max_drift": None,
            "note": "brak sentence-transformers; total_opinion_drift niedostępne",
        }

    by_agent: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        by_agent[entry["agent"]].append((idx, entry["round"]))

    per_agent = {}
    all_drifts = []

    for agent, entries in by_agent.items():
        if len(entries) < 2:
            per_agent[agent] = {
                "drift": None,
                "first_round": entries[0][1] if entries else None,
                "last_round": entries[0][1] if entries else None,
                "note": "za mało wypowiedzi (< 2)",
            }
            continue

        first_idx, first_round = entries[0]
        last_idx, last_round = entries[-1]
        cos_sim = float(np.dot(embeddings[first_idx], embeddings[last_idx]))
        drift = round(1.0 - cos_sim, 4)

        per_agent[agent] = {
            "drift": drift,
            "first_round": first_round,
            "last_round": last_round,
        }
        all_drifts.append((agent, drift))

    mean_drift = _mean([d for _, d in all_drifts])
    max_drift = None
    if all_drifts:
        max_agent, max_val = max(all_drifts, key=lambda x: x[1])
        max_drift = {"agent": max_agent, "value": max_val}

    return {
        "per_agent": {ag: v for ag, v in per_agent.items()},
        "mean": mean_drift,
        "max_drift": max_drift,
    }


def convergence_speed(
    debate_log: list,
    embeddings,
    np,
    *,
    similarity_threshold: float = 0.80,
    delta_threshold: float = 0.03,
) -> dict:
    """
    Ile rund potrzeba żeby between_agent_similarity wzrosła o `delta_threshold`
    względem rundy 1 (metryka relatywna).

    Parametry
    ----------
    similarity_threshold : próg absolutny (zachowany dla wstecznej zgodności)
    delta_threshold      : minimalna zmiana sim względem R1 (domyślnie 0.03)

    Zwraca
    -------
        convergence_round  : pierwsza runda z sim_t − sim_R1 ≥ delta_threshold
        reached            : bool (relatywne kryterium delta)
        reached_absolute   : bool (bezwzgl. sim ≥ similarity_threshold)
        similarity_per_round, delta_per_round, initial_similarity, final_similarity,
        total_delta, threshold, delta_threshold
    """
    if embeddings is None or np is None:
        return {"convergence_round": None, "reached": False,
                "reached_absolute": False,
                "similarity_per_round": {}, "delta_per_round": {},
                "threshold": similarity_threshold,
                "delta_threshold": delta_threshold,
                "note": "brak sentence-transformers; convergence_speed niedostępne"}

    by_round: dict[int, list] = defaultdict(list)
    for idx, entry in enumerate(debate_log):
        by_round[entry["round"]].append(idx)

    sim_per_round = {}
    for rnd, idxs in sorted(by_round.items()):
        if len(idxs) < 2:
            sim_per_round[rnd] = None
            continue
        pairs = [
            float(np.dot(embeddings[i], embeddings[j]))
            for ii, i in enumerate(idxs)
            for j in idxs[ii + 1:]
        ]
        sim_per_round[rnd] = round(_mean(pairs), 4) if pairs else None

    sorted_rnds = sorted(r for r, v in sim_per_round.items() if v is not None)
    initial = sim_per_round[sorted_rnds[0]] if sorted_rnds else None
    final = sim_per_round[sorted_rnds[-1]] if sorted_rnds else None

    delta_per_round = {}
    convergence_round = None
    reached = False
    reached_absolute = False

    for rnd in sorted_rnds:
        sim = sim_per_round[rnd]
        delta = round(sim - initial, 4) if (sim is not None and initial is not None) else None
        delta_per_round[rnd] = delta

        if sim is not None and sim >= similarity_threshold:
            reached_absolute = True

        if (delta is not None and delta >= delta_threshold
                and not reached and rnd != sorted_rnds[0]):
            convergence_round = rnd
            reached = True

    total_delta = round(final - initial, 4) if (final is not None and initial is not None) else None

    return {
        "convergence_round": convergence_round,
        "reached": reached,
        "reached_absolute": reached_absolute,
        "similarity_per_round": sim_per_round,
        "delta_per_round": delta_per_round,
        "initial_similarity": initial,
        "final_similarity": final,
        "total_delta": total_delta,
        "threshold": similarity_threshold,
        "delta_threshold": delta_threshold,
    }


def hedging_rate(
    debate_log: list,
    np=None,
    *,
    mode: str = "auto",
    threshold: float = 0.45,
) -> dict:
    """
    Frakcja zdań zawierających markery niepewności (hedging).

    Tryby: "soft" (embeddingi), "regex", "auto".

    Zwraca
    -------
        per_agent : {agent: {hedged_sentences, total_sentences, hedging_rate}}
        overall   : mean
        mode_used : "soft" | "regex"
    """
    use_soft = False
    if mode in ("soft", "auto") and np is not None:
        centroid = _anchor_centroid(HEDGING_ANCHORS, np)
        use_soft = centroid is not None
    else:
        centroid = None

    by_agent: dict[str, list] = defaultdict(list)
    for entry in debate_log:
        by_agent[entry["agent"]].append(entry["text"])

    per_agent = {}
    all_rates = []

    for agent, texts in by_agent.items():
        combined = " ".join(texts)
        if use_soft:
            matched, total = _soft_match_sentences(combined, centroid, np, threshold)
        else:
            matched, total = _regex_match_sentences(combined, HEDGING_ANCHORS)

        rate = round(matched / total, 4) if total else None
        per_agent[agent] = {
            "hedged_sentences": matched,
            "total_sentences": total,
            "hedging_rate": rate,
        }
        if rate is not None:
            all_rates.append(rate)

    return {
        "per_agent": per_agent,
        "overall": _mean(all_rates),
        "mode_used": "soft" if use_soft else "regex",
        "threshold": threshold if use_soft else None,
    }


def assertiveness_score(
    debate_log: list,
    np=None,
    *,
    mode: str = "auto",
    threshold: float = 0.45,
) -> dict:
    """
    Frakcja zdań zawierających markery pewności/asertywności.

    Zwraca także net_assertiveness = assertiveness_rate - hedging_rate.

    Zwraca
    -------
        per_agent : {agent: {assertive_sentences, total_sentences,
                              assertiveness_rate, hedging_rate, net_assertiveness}}
        overall   : mean assertiveness_rate
        mode_used : "soft" | "regex"
    """
    use_soft = False
    a_centroid = None
    h_centroid = None
    if mode in ("soft", "auto") and np is not None:
        a_centroid = _anchor_centroid(ASSERTIVENESS_ANCHORS, np)
        h_centroid = _anchor_centroid(HEDGING_ANCHORS, np)
        use_soft = a_centroid is not None

    by_agent: dict[str, list] = defaultdict(list)
    for entry in debate_log:
        by_agent[entry["agent"]].append(entry["text"])

    per_agent = {}
    all_rates = []

    for agent, texts in by_agent.items():
        combined = " ".join(texts)
        if use_soft:
            a_match, total = _soft_match_sentences(combined, a_centroid, np, threshold)
            h_match, _ = _soft_match_sentences(combined, h_centroid, np, threshold)
        else:
            a_match, total = _regex_match_sentences(combined, ASSERTIVENESS_ANCHORS)
            h_match, _ = _regex_match_sentences(combined, HEDGING_ANCHORS)

        a_rate = round(a_match / total, 4) if total else None
        h_rate = round(h_match / total, 4) if total else None
        net = round(a_rate - h_rate, 4) if (a_rate is not None and h_rate is not None) else None

        per_agent[agent] = {
            "assertive_sentences": a_match,
            "total_sentences": total,
            "assertiveness_rate": a_rate,
            "hedging_rate": h_rate,
            "net_assertiveness": net,
        }
        if a_rate is not None:
            all_rates.append(a_rate)

    return {
        "per_agent": per_agent,
        "overall": _mean(all_rates),
        "mode_used": "soft" if use_soft else "regex",
        "threshold": threshold if use_soft else None,
    }


def turn_length_entropy(debate_log: list) -> dict:
    """
    Entropia rozkładu długości tokenów per agent (Shannon entropy w bitach).

    Wysoka entropia → agent reaguje adaptacyjnie.
    Niska entropia  → mechaniczna odpowiedź o stałej długości.

    Zwraca
    -------
        per_agent : {agent: {entropy_bits, n_turns, token_lengths, min, max, mean}}
        overall   : mean entropy
    """
    by_agent: dict[str, list] = defaultdict(list)
    for entry in debate_log:
        by_agent[entry["agent"]].append(entry["tokens"])

    per_agent = {}
    all_entropies = []

    for agent, lengths in by_agent.items():
        n = len(lengths)
        if n < 2:
            per_agent[agent] = {
                "entropy_bits": None, "n_turns": n,
                "note": "za mało tur do obliczenia entropii"
            }
            continue

        min_l, max_l = min(lengths), max(lengths)
        n_bins = min(5, n)

        if min_l == max_l:
            entropy = 0.0
        else:
            bin_size = (max_l - min_l) / n_bins
            counts: dict[int, int] = defaultdict(int)
            for l in lengths:
                b = min(int((l - min_l) / bin_size), n_bins - 1)
                counts[b] += 1
            entropy = 0.0
            for count in counts.values():
                p = count / n
                if p > 0:
                    entropy -= p * math.log2(p)
            entropy = round(entropy, 4)

        per_agent[agent] = {
            "entropy_bits": entropy,
            "n_turns": n,
            "token_lengths": lengths,
            "min": min_l,
            "max": max_l,
            "mean": _mean(lengths),
        }
        all_entropies.append(entropy)

    return {
        "per_agent": per_agent,
        "overall": _mean(all_entropies),
    }


def gini_speaking_time(debate_log: list) -> dict:
    """
    Nierówność czasu mówienia między agentami (Gini coefficient tokenów).

    Gini = 0 → równy udział, Gini = 1 → jeden agent monopolizuje debatę.

    Zwraca
    -------
        gini             : Gini coefficient [0, 1]
        per_agent_share  : {agent: frakcja tokenów}
        per_agent_tokens : {agent: total tokens}
        dominant_agent   : agent z największą liczbą tokenów
    """
    by_agent: dict[str, int] = defaultdict(int)
    for entry in debate_log:
        by_agent[entry["agent"]] += entry["tokens"]

    if not by_agent:
        return {"gini": None, "per_agent_share": {}, "per_agent_tokens": {},
                "dominant_agent": None}

    values = sorted(by_agent.values())
    n = len(values)
    total = sum(values)

    if total == 0 or n == 0:
        return {"gini": 0.0, "per_agent_share": {}, "per_agent_tokens": dict(by_agent),
                "dominant_agent": None}

    gini = sum((2 * (i + 1) - n - 1) * values[i] for i in range(n)) / (n * total)
    gini = round(max(0.0, gini), 4)

    per_agent_share = {ag: round(tok / total, 4) for ag, tok in by_agent.items()}
    dominant = max(by_agent, key=by_agent.get)

    return {
        "gini": gini,
        "per_agent_share": per_agent_share,
        "per_agent_tokens": dict(by_agent),
        "dominant_agent": dominant,
    }


# =============================================================================
# Funkcje zbiorcze — wywoływane przez main.py
# =============================================================================

def compute_all(debate_log: list, decision_result: dict, config: dict) -> dict:
    """
    Oblicza wszystkie metryki podstawowe i rozszerzone.

    Poprzednio zwracał tylko metryki z dawnego metrics.py — teraz zawiera
    też wszystkie metryki z dawnego metrics_j.py (jako klucz 'metrics_j').
    Dla wstecznej zgodności z export_csv.py struktura jest zachowana:
        result["metrics"]   — metryki podstawowe
        result["metrics_j"] — metryki rozszerzone
    """
    texts = [e["text"] for e in debate_log]
    embeddings, np_mod = _load_embeddings(texts)
    sentiment_pipe = _load_sentiment_pipeline()

    hist_rep = history_repetition(
        debate_log,
        embeddings=embeddings,
        np=np_mod,
        ngram_threshold=config.get("history_rep_ngram_threshold", 0.25),
        embed_threshold=config.get("history_rep_embed_threshold", 0.85),
        prefix_words=config.get("history_rep_prefix_words", 60),
    )

    cfg = config or {}

    metrics = {
        "tokens_per_turn":                    tokens_per_turn(debate_log),
        "opinion_shift":                      opinion_shift(debate_log, embeddings, np_mod),
        "opinion_shift_sentiment":            opinion_shift_sentiment(debate_log, sentiment_pipe),
        "between_agent_similarity":           between_agent_similarity(debate_log, embeddings, np_mod),
        "between_agent_similarity_sentiment": between_agent_similarity_sentiment(
                                                  debate_log, sentiment_pipe, hist_rep),
        "lexical_richness":                   lexical_richness(debate_log),
        "convergence":                        convergence(decision_result),
        "answer_flip_rate":                   answer_flip_rate(
                                                  debate_log, embeddings, np_mod,
                                                  flip_threshold=cfg.get("flip_threshold", 0.20)),
        "history_repetition":                 hist_rep,
        "semantic_diversity":                 semantic_diversity(debate_log, embeddings, np_mod),
        "redundancy_ratio":                   redundancy_ratio(debate_log, embeddings, np_mod),
    }

    metrics_j = {
        "argument_novelty": argument_novelty(debate_log, embeddings, np_mod),
        "claim_grounding": claim_grounding(
            debate_log, np_mod,
            mode=cfg.get("claim_grounding_mode", "auto"),
            threshold=cfg.get("claim_grounding_threshold", 0.45),
        ),
        "question_density": question_density(debate_log),
        "direct_address_rate": direct_address_rate(debate_log),
        "rebuttal_depth": rebuttal_depth(debate_log, embeddings, np_mod),
        "reciprocal_shift_index": reciprocal_shift_index(debate_log, embeddings, np_mod),
        "opinion_trajectory_monotonicity": opinion_trajectory_monotonicity(
            debate_log, embeddings, np_mod
        ),
        "total_opinion_drift": total_opinion_drift(debate_log, embeddings, np_mod),
        "convergence_speed": convergence_speed(
            debate_log, embeddings, np_mod,
            similarity_threshold=cfg.get("convergence_speed_threshold", 0.80),
            delta_threshold=cfg.get("convergence_speed_delta", 0.03),
        ),
        "hedging_rate": hedging_rate(
            debate_log, np_mod,
            mode=cfg.get("hedging_mode", "auto"),
            threshold=cfg.get("hedging_threshold", 0.45),
        ),
        "assertiveness_score": assertiveness_score(
            debate_log, np_mod,
            mode=cfg.get("assertiveness_mode", "auto"),
            threshold=cfg.get("assertiveness_threshold", 0.45),
        ),
        "turn_length_entropy": turn_length_entropy(debate_log),
        "gini_speaking_time": gini_speaking_time(debate_log),
    }

    return metrics, metrics_j
