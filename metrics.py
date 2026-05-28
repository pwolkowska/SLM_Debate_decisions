"""
metrics.py — Obliczanie metryk z ustrukturyzowanych danych debaty.

7 metryk:
    tokens_per_turn                      — czy osobowość wpływa na ilość produkowanego tekstu?
    opinion_shift                        — semantyczna zmiana stanowiska (cosine distance)
    opinion_shift_sentiment              — zmiana sentymentu runda po rundzie per agent
    between_agent_similarity             — czy agenci produkują różne argumenty i czy się zbliżają?
    between_agent_similarity_sentiment   — porównanie sentymentu między agentami per runda
    lexical_richness                     — czy osobowość wpływa na różnorodność słownictwa?
    convergence                          — czy debata kończy się porozumieniem i jak szybko?
    answer_flip_rate                     — NoF i ToF (sycophancy / position stability)
    history_repetition                   — czy agent zaczyna od powtarzania historii debaty?

Zależności opcjonalne:
    sentence-transformers  →  opinion_shift, between_agent_similarity, answer_flip_rate
    transformers           →  opinion_shift_sentiment, between_agent_similarity_sentiment

Źródła:
    Hong et al. 2025, SYCON Bench (arXiv:2505.23840)
    Laban et al. 2023, FlipFlop (arXiv:2311.08596)
"""

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
    Można zastąpić dowolnym innym modelem wielojęzycznym.
    """
    try:
        from transformers import pipeline
        sentiment = pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None,           # zwraca wszystkie klasy z prawdopodobieństwami
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
    """Zwraca pierwsze `words` słów tekstu (to co agent mówi na początku wypowiedzi)."""
    return " ".join(text.split()[:words])

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
    Sprawdza, czy agent zaczyna wypowiedź od powtórzenia historii debaty
    (dosłownie lub parafrazując).

    Dla każdej wypowiedzi (poza pierwszą rundą) porównuje prefix agenta
    z wszystkimi poprzednimi wypowiedziami:

      - ngram_overlap  ≥ ngram_threshold  → powtórzenie dosłowne / niemal dosłowne
      - cosine_sim     ≥ embed_threshold  → powtórzenie semantyczne (wymaga embeddings)

    Parametry
    ----------
    ngram_threshold : próg Jaccard n-gram (domyślnie 0.25)
    embed_threshold : próg cosine similarity (domyślnie 0.85)
    prefix_words    : ile słów liczymy jako "początek" wypowiedzi (domyślnie 60)
    ngram_n         : rząd n-gramów (domyślnie 3)

    Zwraca
    -------
    dict z kluczami:
        per_entry : lista dict {round, agent, ngram_hit, embed_hit, max_ngram, max_cosine}
        per_agent : {agent: {repetition_count, total_turns, repetition_rate}}
        summary   : {total_repetitions, total_turns, overall_rate}
    """
    per_entry = []
    history_texts: list[str] = []  # kumulowana historia tekstów (poprzednie wypowiedzi)
    history_indices: list[int] = []  # indeksy w debate_log

    for idx, entry in enumerate(debate_log):
        current_prefix = _prefix(entry["text"], prefix_words)

        if not history_texts:
            # Pierwsza wypowiedź — brak historii do porównania
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

        # --- n-gram overlap ---
        ngram_scores = [
            _ngram_overlap(current_prefix, hist, ngram_n)
            for hist in history_texts
        ]
        max_ngram = round(max(ngram_scores), 4)
        ngram_hit = max_ngram >= ngram_threshold

        # --- cosine similarity (opcjonalne) ---
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

    # Agregacja per agent
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
    total_reps = sum(
        1 for e in per_entry if e["ngram_hit"] or e["embed_hit"]
    )

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

    Dla każdego agenta liczy skalar sentymentu [-1, +1] per wypowiedź,
    następnie:
        - trajectory     : lista (runda, sentyment) w kolejności wystąpień
        - delta_first_last: sentyment_ostatni − sentyment_pierwszy
        - mean_shift_abs  : średnia |Δsentyment| między kolejnymi wypowiedziami

    Wyższy |delta_first_last| = większa zmiana postawy emocjonalnej.

    Wymaga: transformers (pip install transformers)
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


#

def between_agent_similarity_sentiment(
    debate_log: list,
    sentiment_pipeline,
    history_rep_result: Optional[dict] = None,
) -> dict:
    """
    Porównuje sentyment między agentami w każdej rundzie.

    Dla każdej rundy:
        - liczy sentyment każdego agenta
        - oblicza średnią różnicę bezwzględną |s_i − s_j| dla każdej pary
          (niski wynik = agenci zbliżeni emocjonalnie, wysoki = różnią się)
        - jeśli przekazano history_rep_result, oznacza wypowiedzi będące powtórzeniem

    Parametry
    ----------
    history_rep_result : wynik funkcji history_repetition(); jeśli podany,
                         wypowiedzi zidentyfikowane jako powtórzenia są oznaczane
                         flagą `is_repetition` i opcjonalnie wykluczone ze średniej.

    Zwraca
    -------
        per_round    : {runda: {agent: sentiment, "mean_pair_diff": float, "agents_with_repetition": list}}
        initial      : mean_pair_diff rundy 1
        final        : mean_pair_diff ostatniej rundy
        delta        : final − initial (>0 = agenci coraz bardziej różnią się sentymentem)
        note         : komunikat gdy brak zależności
    """
    if sentiment_pipeline is None:
        return {
            "per_round": {},
            "initial": None,
            "final": None,
            "delta": None,
            "note": "brak transformers lub nie udało się załadować modelu sentymentu",
        }

    # Zbuduj mapę: (runda, agent) → is_repetition
    rep_map: dict[tuple, bool] = {}
    if history_rep_result:
        for e in history_rep_result.get("per_entry", []):
            rep_map[(e["round"], e["agent"])] = e["ngram_hit"] or e["embed_hit"]

    # Zbierz teksty per runda
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

        # Pary agentów — różnica sentymentu
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


def _position_label(text: str, embeddings, np, all_texts: list, all_labels: list) -> int:
    """
    Przypisuje wypowiedź do klastra pozycji (0 lub 1) na podstawie cosine similarity
    do centroidów. Używane wewnętrznie przez answer_flip_rate.

    Jeśli embeddings niedostępne, wraca -1 (nieznane).
    """
    # Brak embeddingów — fallback niedostępny
    if embeddings is None:
        return -1
    raise NotImplementedError("Używaj _cluster_positions zamiast tej funkcji.")


def _cosine_sim(emb_a, emb_b, np) -> float:
    return float(np.dot(emb_a, emb_b))


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

    Wykrywanie zmiany pozycji:
        Używamy cosine distance między kolejnymi wypowiedziami agenta.
        Jeśli 1 − cosine_sim(t, t−1) ≥ flip_threshold → flip.

        flip_threshold = 0.20 oznacza ~20% zmianę w przestrzeni semantycznej.
        (Wartość empiryczna; można dostroić przez config.)

    Jeśli brak embeddings — zwraca notę z informacją o brakującej zależności.

    Parametry
    ----------
    flip_threshold : próg odległości cosinus sygnalizujący zmianę pozycji (domyślnie 0.20)

    Zwraca
    -------
        per_agent : {agent: {nof, tof, flip_rounds, trajectory}}
        summary   : {mean_nof, max_nof, agents_with_flip, flip_threshold}
        note      : komunikat gdy brak embeddings
    """
    if embeddings is None or np is None:
        return {
            "per_agent": {},
            "summary": {},
            "note": "brak sentence-transformers; NoF/ToF niedostępne",
        }

    # Zbierz wypowiedzi per agent w kolejności rund
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

        # Pierwsza wypowiedź — punkt odniesienia
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
    """1 − mean(pairwise cos_sim) dla wszystkich par wypowiedzi debaty.
    Uzupełnienie between_agent_similarity: tamta mierzy per runda,
    ta mierzy globalnie (funneling effect całej debaty).
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
    """Frakcja wypowiedzi w rundzie t semantycznie podobnych (cos_sim > threshold)
    do jakiejkolwiek wypowiedzi z wcześniejszych rund.
    0.0 = każda wypowiedź jest nowa, 1.0 = debata kręci się w kółko.
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
# Funkcja zbiorcza — wywoływana przez main.py
# =============================================================================

def compute_all(debate_log: list, decision_result: dict, config: dict) -> dict:
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

    return {
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
                                                  flip_threshold=config.get("flip_threshold", 0.20)),
        "history_repetition":                 hist_rep,
        # nowe
        "semantic_diversity":                 semantic_diversity(debate_log, embeddings, np_mod),
        "redundancy_ratio":                   redundancy_ratio(debate_log, embeddings, np_mod),
    
    }