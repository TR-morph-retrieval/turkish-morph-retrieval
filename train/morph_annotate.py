#!/usr/bin/env python
"""Morphological annotation + variant groups for the generated dataset. Zero API calls.

    conda run -n dl_hw1 python morph_annotate.py --split train --split dev

Adds two things to every item, written back in place (`item["morphology"]`, `item["variants"]`):

**Morphological annotation**, two tiers, each tagged with its own confidence:

- Tier 1 (deterministic, no dependency): recovers `stem` / `diff_query` / `diff_counterfactual` /
  `shared_tail` for the critical-word pair by a *windowed* boundary search — not the raw
  longest-common-prefix `morph_validators.soft_lcp_len` uses, which over-extends the stem when the
  boundary character repeats (`yaptırmamıştı`/`yaptırmıştı`: raw LCP puts the cut after the extra
  `m` that actually belongs to `-mış-`, giving diff `am`/`` instead of the true `ma`/``). The search
  tries small shifts around the naive cut and keeps whichever gives an EXACT hit against
  `ALLOMORPH_TABLE` (parsed from `morph_taxonomy.TARGET_FEATURES`'s own `ek_turu` strings, not
  hand-written). Falls back to the raw cut, tagged `match: skeleton`, when nothing matches exactly.
- Tier 2 (`zeyrek`, optional): full-word lemma + gold morpheme tags for the critical-word pair.
  Degrades to Tier-1-only if `zeyrek` is not installed — checked once at import, never raises later.

The search runs against the FULL allomorph table, not just the item's own declared
`target_feature` — deliberately, because that is what turns this into a label audit rather than a
confirmation exercise. `almamışlardı`/`almışlardı` is labelled `POSS.PL.ABL` in the data but the
actual textual contrast is negation; matching against the full table finds `NEG`, matching against
only the declared feature's forms would not, and it is exactly this kind of item the agreement-rate
statistic is meant to surface.

**Variant groups**: orthographic (Turkish-deasciified query + every candidate, `İ`/`I`-correct) and
lemma-family (positive/counterfactual grouped by zeyrek lemma when available, else by the
deterministic stem) — the latter is what enables a root-family retrieval diagnostic: same root
should cluster, one suffix apart should separate, and a channel that only does one of those is
failing the other half of the task.
"""
import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import morph_validators as V
from morph_taxonomy import TARGET_FEATURES

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_morph_v2"

_ZEYREK_INIT_ERROR = None
try:
    import zeyrek
    # zeyrek's rule-based analyzer logs every candidate parse path at WARNING level (its own
    # library bug — this is debug-level tracing, not a warning) — one line per ambiguous parse,
    # so a 600-item dataset floods stdout with thousands of "APPENDING RESULT: ..." lines.
    logging.getLogger("zeyrek").setLevel(logging.ERROR)
    _cand = zeyrek.MorphAnalyzer()
    _ = _cand.analyze("kitap")
    _ANALYZER = _cand
except Exception as _exc:                                  # noqa: BLE001 - optional dependency
    _ANALYZER = None
    _ZEYREK_INIT_ERROR = f"{type(_exc).__name__}: {_exc}"

# --------------------------------------------------------------------------- allomorph table
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_FORM_RE = re.compile(r"-?([a-zçğıöşü]+)")                 # one dash-prefixed Turkish suffix form


def _parse_allomorph_forms(ek_turu):
    """Pull surface suffix forms out of an `ek_turu` string like 'olumsuzluk (-ma/-me)'.

    Only the LAST parenthetical is used: several `ek_turu` strings carry an earlier gloss
    parenthetical too (none currently do, but this keeps the parser from misreading one if it
    shows up), and the suffix forms are always the final group by construction in
    `morph_taxonomy.py`. Returns [] if nothing dash-prefixed is found there — most chain-feature
    glosses are plain language with no parenthetical at all.
    """
    parens = _PAREN_RE.findall(ek_turu)
    if not parens:
        return []
    forms = []
    for piece in re.split(r"[/,]", parens[-1]):
        m = _FORM_RE.match(piece.strip())
        if m:
            forms.append(m.group(1))
    return forms


def build_allomorph_table():
    """{feature_key: [surface forms]}, direct from the taxonomy — not a hand-maintained list.

    Includes every feature (`layer` single or chain): a chain feature's own fused surface form
    (`CNTR` -> `seydi`) is a legitimate direct hit even though the feature is a composition, and is
    tried before the code falls back to decomposing the key into single-layer constituents.
    """
    table = {}
    for f in TARGET_FEATURES:
        forms = _parse_allomorph_forms(f["ek_turu"])
        if forms:
            table[f["key"]] = forms
    return table


ALLOMORPH_TABLE = build_allomorph_table()
_SINGLE_LAYER_KEYS = {f["key"] for f in TARGET_FEATURES if f["layer"] == "single"}


def _decompose(feature_key):
    """Chain key -> its single-layer constituent keys, by greedy longest dot-segment match.

    `CAUS.PASS.NEG` -> [CAUS, PASS, NEG] (full resolution: all three are single-layer keys).
    `POSS.PL.ABL` -> [PL, ABL] (partial: `POSS` alone names no feature — Turkish possessive person
    isn't recoverable from the key text — so that segment is silently dropped rather than guessed).
    Best-effort by design; callers must not assume completeness.
    """
    segs = feature_key.split(".")
    out, i = [], 0
    while i < len(segs):
        hit = None
        for j in range(len(segs), i, -1):
            cand = ".".join(segs[i:j])
            if cand in _SINGLE_LAYER_KEYS:
                hit = cand
                i = j
                break
        if hit:
            out.append(hit)
        else:
            i += 1
    return out


# Buffer consonants Turkish inserts at a vowel/vowel or vowel-initial-suffix boundary, and the
# devoicing a suffix-initial stop shows after a voiceless stem-final. Stripping these before
# comparing a candidate diff region to a table form is what lets `araba`+DAT `-a` match the real
# surface `arabaya`, and `kitap`+LOC `-de` match the real surface `kitapta`.
#
# Reuses `morph_validators`'s already-validated devoicing map (`_SOFTEN`, voiced->voiceless:
# b/c/d/ğ -> p/ç/t/k) rather than a second hand-written one — an earlier version of this file had
# its own `_DEVOICE` dict checking the WRONG direction (only matched a form already spelled
# voiceless, never converted a voiced citation form like 'diğ' to its voiceless surface 'tığ'),
# which silently broke every own-feature match for forms with a devoicing-eligible first
# consonant. Confirmed on real data: NMLZ.DIK's registered form is 'diğ' (voiced citation form),
# but `başlattığı` realises it as 'tığ' (devoiced AND back-harmonized) — with the old code neither
# _DEVOICE nor any vowel-harmony variant existed, so this — one of the most common Turkish
# participle surface forms — silently fell through to a spurious global cross-key match instead
# (misreported as EQU->ACC) across every affected item in the dataset, not just this one.
_BUFFERS = "ynsş"


def _harmony_variants(form):
    """A table form's vowel-harmony paradigm: every vowel position varies TOGETHER (Turkish
    suffix vowels within one bound morpheme harmonize as a unit, not independently), keyed by
    whether that position is a HIGH vowel (4-way ı/i/u/ü) or a LOW vowel (2-way a/e). Needed
    because several `ek_turu` entries record only ONE harmony variant (`NMLZ.DIK`: 'diğ', never
    'dığ'/'duğ'/'düğ') where others spell out the full set explicitly (`ACC`: i/ı/u/ü already all
    four) — this generalises the under-specified ones instead of special-casing them by hand.
    """
    idx_high = [i for i, ch in enumerate(form) if ch in V.HIGH]
    idx_low = [i for i, ch in enumerate(form) if ch in V.VOWELS and ch not in V.HIGH]
    if not idx_high and not idx_low:
        return {form}
    high_opts = ["ı", "i", "u", "ü"] if idx_high else [None]
    low_opts = ["a", "e"] if idx_low else [None]
    out = set()
    for h in high_opts:
        for lo in low_opts:
            chars = list(form)
            for i in idx_high:
                chars[i] = h
            for i in idx_low:
                chars[i] = lo
            out.add("".join(chars))
    return out


def _form_variants(form):
    """A table form, plus its vowel-harmony, buffer-consonant and devoicing variants."""
    variants = set()
    for base in _harmony_variants(form):
        variants.add(base)
        if base:
            # Devoicing alternates only the SUFFIX-INITIAL consonant (the one adjacent to the
            # stem) — never an internal or final one belonging to the morpheme itself.
            # `str.translate` over the whole string was an earlier bug here: 'dığ'.translate(...)
            # devoiced BOTH the leading 'd' (correct: -> 't') and the trailing 'ğ' (wrong: -> 'k'),
            # producing the nonexistent 'tık' instead of the real surface form 'tığ'.
            devoiced0 = base[0].translate(V._SOFTEN)
            if devoiced0 != base[0]:
                variants.add(devoiced0 + base[1:])
    for buf in _BUFFERS:
        for base in list(variants):
            variants.add(buf + base)
    return variants


def _search_boundaries(a, b, window=3):
    """Yield (p, s, a_mid, b_mid) for cuts near the naive shared-prefix/suffix boundary.

    `p` = prefix length, `s` = suffix length, recomputed FRESH at each `p` (not reused globally) so
    a shift in `p` can never make `p + s` exceed either word's length — that overlap is exactly what
    over-extended the naive LCP-only cut. `a_mid`/`b_mid` are what's left in between: the candidate
    diff region for each word.

    At each `p`, `s` ranges from the maximal common suffix DOWN to 0 (max first, so callers that
    want a single default — the 'skeleton' fallback — get the same greedy cut as before). Trying
    only the maximal `s` was an earlier gap: a two-morpheme diff region where just the OUTER
    morpheme happens to be identical on both sides (`-diğ-i` vs `-tığ-ı`, both ending in the
    possessive `-ı`) greedily absorbs that shared `ı` into the tail, leaving `a_mid`/`b_mid` short
    one character on each side and never matching either registered form. Trying smaller `s` too
    lets p=6,s=1 surface `'acağ'`/`'tığ'` — the two forms actually in the table — instead of only
    ever seeing p=6,s=2's `'aca'`/`'tı'`, which are neither.

    `p` is capped at `lcp`, never `lcp + window`: `lcp` is where `a` and `b` stop being
    (softening-)equivalent, so `a[:p]` for any `p > lcp` is not a prefix `b` actually shares —
    `stem=a[:p]` would then be returned as if both words started with it when they provably don't
    (confirmed: `güncelletsin`/`güncelletti` have lcp=9, but the unclamped window let p reach 10,
    handing back stem=`'güncellets'` — a string `güncelletti` does not start with). The window is
    only ever useful shrinking p BELOW lcp, to split a shared run into an inner boundary; growing
    it past lcp can only fabricate a match, never find a real one.
    """
    lcp = V.soft_lcp_len(a, b)
    lo, hi = max(0, lcp - window), min(len(a), len(b), lcp)
    for p in range(lo, hi + 1):
        max_s = V.common_suffix_len(a[p:], b[p:])
        for s in range(max_s, -1, -1):
            a_mid, b_mid = a[p:len(a) - s], b[p:len(b) - s]
            yield p, s, a_mid, b_mid


# Precomputed once at import (ALLOMORPH_TABLE is fixed): `_form_variants` was being rebuilt from
# scratch — harmony expansion, devoicing, buffer-prefixing — on every single (boundary, key) pair
# `find_boundary_match` tests, which is the same ~40-variant computation repeated thousands of
# times per item. Measured impact: 37ms/call before caching, sub-millisecond after.
_KEY_VARIANTS = {key: {v for f in forms for v in _form_variants(f)}
                 for key, forms in ALLOMORPH_TABLE.items()}


def _key_hit(mid, key):
    return mid in _KEY_VARIANTS[key] or mid == ""


def find_boundary_match(a, b, own_feature=None):
    """Search for an exact allomorph-table hit near the naive stem/suffix boundary of (a, b).

    Four match kinds, tried in this priority order (measured on v2.2: restricting to same-key
    matching alone left 63% of single-layer items as 'skeleton' — PROP('li')/PRIV('sız') and
    LOC('de')/ABL('den') both fail same-key matching not because the text is wrong but because
    the counterfactual legitimately realises a DIFFERENT declared feature's own suffix, not an
    allomorph of the query's feature):

    1. own-feature, same key    — a_mid AND b_mid both match one of own_feature's own forms.
       Strongest possible evidence: the item's own label predicts both surface forms exactly.
    2. own-feature, cross key   — ONE side matches an own_feature form, the OTHER matches some
       DIFFERENT key's form. Covers PROP/PRIV, LOC/ABL: the item's label is trusted for the side
       that matches it, and the other side is accepted as a real (if differently-keyed) morpheme
       rather than demanded to be an allomorph of the same key.
    3. global, same key         — the label-AUDIT path: neither side needs to match own_feature,
       but both sides DO match some other single key. This is how `almamışlardı`/`almışlardı`
       (labelled POSS.PL.ABL) is found to actually be NEG.
    4. global, cross key        — both sides match a table form, but of two DIFFERENT keys, and
       neither is the item's own declared feature. Weakest evidence kept; every side (own or not)
       is held to the >=2-char noise floor here, since nothing anchors the match to this item.

    Returns dict(stem, diff_query, diff_counterfactual, shared_tail, canonical_morpheme, match,
    match_source) — `match` is 'exact' | 'skeleton', `canonical_morpheme` is a single key for a
    same-key hit or "KEY_A->KEY_B" for a cross-key hit. Always returns a structurally valid
    decomposition (stem+diff+tail reconstructs each input exactly) even with no table match.
    """
    own_keys = ([own_feature] if own_feature in ALLOMORPH_TABLE else []) + _decompose(own_feature or "")
    own_keys = [k for k in dict.fromkeys(own_keys) if k in ALLOMORPH_TABLE]     # dedupe, keep order
    all_keys = list(ALLOMORPH_TABLE)

    RANK = {"own_same": 0, "own_cross": 1, "global_same": 2, "global_cross": 3}
    best = None            # (rank, size, p, s, a_mid, b_mid, morpheme)

    def consider(rank_name, p, s, a_mid, b_mid, morpheme):
        # Within a rank tier, prefer the LARGER total diff, not the smaller. A short diff is
        # LESS specific, not more: single characters like 'e' or 'n' are shared by many unrelated
        # forms (OPT's bare '-e', PTCP.SUBJ's bare '-n', PASS's bare '-n', ...) purely by
        # coincidence, so preferring them systematically out-competes real, longer, far less
        # coincidental matches. Concretely: 'evde'/'evden' has both a 1-char match ('e'/'en' ->
        # OPT/PTCP.SUBJ, pure accident) and the correct 2/3-char one ('de'/'den' -> LOC/ABL) in the
        # same rank tier; preferring the shorter one picked the accident every time until fixed.
        nonlocal best
        cand = (RANK[rank_name], -(len(a_mid) + len(b_mid)), p, s, a_mid, b_mid, morpheme)
        if best is None or cand[:2] < best[:2]:
            best = cand

    for p, s, a_mid, b_mid in _search_boundaries(a, b):
        if not (a_mid or b_mid):
            continue
        # 1/2: own-feature anchored. A bare single-consonant allomorph (PASS/REFL's -n/-ş) is
        # trusted at length 1 ONLY on the side that matches the item's own claimed feature — the
        # item asserting "this tests X" is exactly the context in which a minimal bare hit for X
        # is plausible rather than coincidental.
        for ok in own_keys:
            # Genuine (non-empty) evidence only for the OWN anchor. `_key_hit(mid, key)` treats
            # mid == "" as a trivial hit for EVERY key (needed for own_same, where a real bare
            # allomorph like PASS's -n vs empty is the whole point) — but that means an empty
            # b_mid would "belong" to whichever own_key happens to be checked first, regardless of
            # which one is actually relevant. Confirmed case: CAUS.PASS.NEG's own_keys iterate
            # CAUS before NEG; imzalatılamamıştı/imzalatılmıştı has b_mid='' and a_mid='ama' (a
            # real NEG.ABIL hit), and without this guard CAUS — which matched nothing — won the
            # cross-key label purely because it happened to be checked first against an empty
            # string. `a_own`/`b_own` (used for own_same, where symmetry is exactly the point)
            # keep the permissive empty-hit rule; only the ANCHOR side of a cross-key claim below
            # is required to be real.
            a_own, b_own = _key_hit(a_mid, ok), _key_hit(b_mid, ok)
            if a_own and b_own:
                consider("own_same", p, s, a_mid, b_mid, ok)
            elif a_mid and _key_hit(a_mid, ok) and b_mid and max(len(b_mid), 1) >= 2:
                other = next((k for k in all_keys if k != ok and _key_hit(b_mid, k)), None)
                if other:
                    consider("own_cross", p, s, a_mid, b_mid, f"{ok}->{other}")
            elif b_mid and _key_hit(b_mid, ok) and a_mid and max(len(a_mid), 1) >= 2:
                other = next((k for k in all_keys if k != ok and _key_hit(a_mid, k)), None)
                if other:
                    consider("own_cross", p, s, a_mid, b_mid, f"{other}->{ok}")
        # 3/4: global label-audit. Both sides held to the 2-char floor regardless of key, per the
        # existing rationale (a single coincidental consonant like the 'n' in evde/evden is not
        # specific enough to count as evidence when nothing else anchors the match to this item).
        nonempty_len = max(len(a_mid), len(b_mid))
        if nonempty_len >= 2:
            same = next((k for k in all_keys if _key_hit(a_mid, k) and _key_hit(b_mid, k)), None)
            if same:
                consider("global_same", p, s, a_mid, b_mid, same)
            else:
                ka = next((k for k in all_keys if _key_hit(a_mid, k)), None)
                kb = next((k for k in all_keys if _key_hit(b_mid, k)), None)
                if ka and kb and ka != kb:
                    consider("global_cross", p, s, a_mid, b_mid, f"{ka}->{kb}")
        if best and best[0] == 0:
            break           # an own-feature same-key hit at this boundary is as good as it gets

    if best:
        _, _, p, s, a_mid, b_mid, morpheme = best
        return dict(stem=a[:p], diff_query=a_mid, diff_counterfactual=b_mid, shared_tail=a[len(a) - s:],
                   canonical_morpheme=morpheme, match="exact")

    p0, s0, a_mid0, b_mid0, *_ = next(iter(_search_boundaries(a, b, window=0)))
    return dict(stem=a[:p0], diff_query=a_mid0, diff_counterfactual=b_mid0, shared_tail=a[len(a) - s0:],
               canonical_morpheme=None, match="skeleton")


# --------------------------------------------------------------------------- zeyrek (Tier 2)
# Zeyrek/TRmorph tag -> our feature key. Hand-authored: entries marked (observed) were confirmed
# directly against zeyrek's actual output during development; the rest are well-established
# extensions of an observed pattern (e.g. P1pl observed -> P2sg/P3pl follow the same naming). A
# wrong entry here only REDUCES coverage (the mapped feature just never matches), it cannot
# produce a false agreement, so an incomplete table is safe to ship rather than a correctness risk.
ZEYREK_TAG_MAP = {
    "Neg": "NEG", "Unable": "NEG.ABIL", "Able": "ABIL",                      # observed
    "Caus": "CAUS", "Pass": "PASS", "Recip": "RECP", "Reflex": "REFL",       # Caus/Pass observed
    "Past": "PST", "Narr": "PRF.EVID", "Prog1": "PRS.PROG", "Prog2": "PRS.PROG",
    "Fut": "FUT", "Aor": "AOR",                                              # Past/Narr/Aor observed
    "Neces": "NEC", "Desr": "COND", "Opt": "OPT", "Imp": "IMP.3",            # Neces/Desr observed
    "Cond": "COND",
    "P1sg": "POSS.1SG", "P2sg": "POSS.2SG", "P3sg": "POSS.3SG",              # P1pl/P3sg observed
    "P1pl": "POSS.1PL", "P2pl": "POSS.2PL", "P3pl": "POSS.3PL",
    "A3pl": "PL",
    "Dat": "DAT", "Loc": "LOC", "Abl": "ABL", "Acc": "ACC",                  # Dat/Loc/Abl observed
    "Ins": "INS", "Gen": "GEN", "Equ": "EQU",
    "With": "PROP", "Without": "PRIV", "Rel": "REL.KI",
    "Agt": "AGT", "Ness": "ABST", "Become": "VBLZ", "Acquire": "VBLZ",
    "Distrib": "DISTR",
    "Inf1": "NMLZ.MEK", "Inf2": "NMLZ.ME",                                   # observed
    "AfterDoingSo": "CVB.AND", "When": "CVB.WHEN", "While": "CVB.WHILE",     # observed
    "WithoutHavingDoneSo": "CVB.WITHOUT", "SinceDoingSo": "CVB.SINCE",
    "AsLongAs": "CVB.ASLONG", "ByDoingSo": "CVB.BY",
    "PastPart": "PTCP.OBJ", "FutPart": "PTCP.FUT", "PresPart": "PTCP.SUBJ",
}


def _zeyrek_parse(word):
    """One word -> (lemma, morphemes) or None if unavailable/unparsed. Never raises."""
    if _ANALYZER is None or not word or " " in word:
        return None
    try:
        parses = _ANALYZER.analyze(word)[0]
    except Exception:                                       # noqa: BLE001 - never let this crash a run
        return None
    if not parses or parses[0].lemma == "Unk" or not isinstance(parses[0].morphemes, list):
        return None
    p = parses[0]
    return p.lemma, p.morphemes


# KNOWN CONFOUND on zeyrek_target_feature_agreement (measured: 40.2% agree despite a 90.8% parse
# rate — lower than parse rate alone would suggest, and NOT a zeyrek or Tier-1 defect). Root cause
# lives in `morph_validators.resolve_critical_pair`, not here: for chain features whose reported
# `critical_word_query`/`_counterfactual` is a PHRASE (rejected by the single-word check), it falls
# back to `derive_critical_pair(positive, counterfactual)` first and RETURNS on the first pair that
# shares a stem — even an incidental one. Confirmed case (evid_cond_neg_0690_361d05, reported
# 'duyurmuş olsaydı'/'duyurmuştu' — genuinely the evidential+conditional+negation contrast): the
# independently-reworded positive text contains an unrelated noun in a different case
# ('biletinin' vs the counterfactual's 'biletini'), which shares a stem and is accepted before the
# query-vs-counterfactual fallback — which WOULD have recovered the true verbal contrast — is ever
# tried. Both tiers inherit this since both call the same `resolve_critical_pair`. Not fixed here:
# it is existing, self-tested pipeline code with its own calibration against v1.3.1, and changing
# the fallback ORDER needs its own measurement pass, not a change bundled into an annotation tool.
def tier2_annotate(word_query, word_cf):
    if _ANALYZER is None:
        return None
    pq, pc = _zeyrek_parse(word_query), _zeyrek_parse(word_cf)
    out = {
        "query_lemma": pq[0] if pq else None, "query_morphemes": pq[1] if pq else None,
        "counterfactual_lemma": pc[0] if pc else None, "counterfactual_morphemes": pc[1] if pc else None,
    }
    if pq and pc:
        diff = sorted(set(pq[1]) ^ set(pc[1]))               # symmetric difference: what changed
        out["tag_diff"] = diff
        mapped = {ZEYREK_TAG_MAP[t] for t in diff if t in ZEYREK_TAG_MAP}
        out["mapped_feature"] = sorted(mapped) or None
    else:
        out["tag_diff"] = None
        out["mapped_feature"] = None
    return out


# --------------------------------------------------------------------------- variant groups
# What a user actually types: no diacritics, and — since this is compared against tr_lower'd query
# text elsewhere — already lowercase. Applied AFTER tr_lower so 'İ'/'I' are resolved correctly
# first; a plain str.translate on the original casing would mis-map 'İ' the same way str.lower() did.
_DEASCII = str.maketrans("çğıöşü", "cgiosu")


def deascii(text):
    return V.tr_lower(text).translate(_DEASCII)


def _agrees_with_target(morpheme, target_feature):
    """True if `morpheme` (a single key, or 'A->B' for a cross-key hit) names `target_feature`
    on either pole. A cross-key hit is "about" target_feature if the item's own declared feature
    shows up on EITHER side of the pair — that is the whole point of resolving it as A->B rather
    than discarding it, e.g. a PROP item realised as PRIV->PROP still confirms the PROP label."""
    if not morpheme:
        return False
    keys = morpheme.split("->")
    decomp = set(_decompose(target_feature or ""))
    return any(k == target_feature or k in decomp for k in keys)


# --------------------------------------------------------------------------- per-item annotation
def annotate_item(item):
    resolved = V.resolve_critical_pair(item)
    if resolved is None:
        # No verified pair exists: `resolve_critical_pair` already tried the reported fields (only
        # if they check out against the actual texts) AND derivation from positive/query vs the
        # counterfactual, and NONE recovered a coherent shared-stem pair. Running the boundary
        # search on the raw, unverified `critical_word_*` fields anyway would produce a diff region
        # between two unrelated strings and present it as if it meant something — worse than no
        # annotation. This is a distinct, more serious status than 'skeleton' (which does have a
        # real stem+diff, just no table hit): 'no_pair' means the item itself has no recoverable
        # word-level contrast, which on inspection has caught genuinely broken generated candidates
        # (see generation_report.md's morphology_annotation.no_pair_examples).
        morphology = dict(stem=None, diff_query=None, diff_counterfactual=None, shared_tail=None,
                          canonical_morpheme=None, match="no_pair", target_feature_agrees=None,
                          reported_query=item.get("critical_word_query"),
                          reported_counterfactual=item.get("critical_word_counterfactual"))
        w_q = w_cf = None
    else:
        w_q, w_cf = resolved[0], resolved[1]
        morphology = find_boundary_match(w_q, w_cf, own_feature=item.get("target_feature"))
        morphology["target_feature_agrees"] = (
            _agrees_with_target(morphology["canonical_morpheme"], item.get("target_feature"))
            if morphology["match"] == "exact" else None)

    z = tier2_annotate(w_q, w_cf)
    if z is not None:
        if z["mapped_feature"]:
            z["agrees_with_target_feature"] = any(
                _agrees_with_target(mf, item.get("target_feature")) for mf in z["mapped_feature"])
            t1_keys = set((morphology["canonical_morpheme"] or "").split("->")) - {""}
            z["agrees_with_tier1"] = bool(t1_keys & set(z["mapped_feature"]))
        else:
            z["agrees_with_target_feature"] = None
            z["agrees_with_tier1"] = None
        morphology["zeyrek"] = z

    lemma_family = {"query": None, "positive": None, "counterfactual": None}
    if z:
        lemma_family["query"] = z.get("query_lemma")
        lemma_family["counterfactual"] = z.get("counterfactual_lemma")
    if not lemma_family["query"]:
        # Tier-1 fallback: the derived shared stem stands in for a lemma when zeyrek is unavailable
        # or fails to parse — coarser (no vowel-drop/consonant-softening normalisation) but always
        # available, and still enough to cluster forms of one root together.
        lemma_family["query"] = lemma_family["counterfactual"] = morphology["stem"] or None

    variants = {
        "query": deascii(item["query"]),
        "candidates": {c["id"]: deascii(c["text"]) for c in item["candidates"]},
        "lemma_family": lemma_family,
    }
    return morphology, variants


def annotate_dataset(items, use_zeyrek=True):
    global _ANALYZER
    if not use_zeyrek:
        _ANALYZER = None

    stats = Counter()
    tf_agree = Counter()          # exact-match items only: True / False
    zy_tf_agree = Counter()
    zy_t1_agree = Counter()
    no_pair_examples = []
    disagreement_examples = []
    for it in items:
        morphology, variants = annotate_item(it)
        it["morphology"] = morphology
        it["variants"] = variants
        stats["total"] += 1
        stats[f"match_{morphology['match']}"] += 1
        if (morphology["match"] == "exact" and morphology["target_feature_agrees"] is False
                and len(disagreement_examples) < 20):
            # The label-audit path's actual output: Tier 1 found a clean, exact morpheme match —
            # just not the one the item claims to test. `almamışlardı`/`almışlardı` (labelled
            # POSS.PL.ABL, textually NEG) is the case that motivated tracking this at all.
            disagreement_examples.append({
                "query_id": it["query_id"], "declared": it["target_feature"],
                "found": morphology["canonical_morpheme"],
                "diff": f"{morphology['diff_query']!r}/{morphology['diff_counterfactual']!r}"})
        if morphology["match"] == "no_pair" and len(no_pair_examples) < 20:
            # Kept as a first-class report output, not swept into an aggregate count: every one
            # found during development was either a genuinely broken generated candidate (e.g. a
            # nonsense word) or a mislabeled item (declared feature realised as a suppletive
            # lexical pair, not a suffix) that slipped past the judge — worth a human look, not
            # something the annotator should paper over.
            no_pair_examples.append({"query_id": it["query_id"], "target_feature": it["target_feature"],
                                     "reported_query": morphology["reported_query"],
                                     "reported_counterfactual": morphology["reported_counterfactual"]})
        if morphology["match"] == "exact":
            tf_agree[morphology["target_feature_agrees"]] += 1
        z = morphology.get("zeyrek")
        if z and z.get("query_morphemes") is not None:
            stats["zeyrek_parsed"] += 1
            if z["mapped_feature"]:
                zy_tf_agree[z["agrees_with_target_feature"]] += 1
                zy_t1_agree[z["agrees_with_tier1"]] += 1

    n = stats["total"] or 1
    report = {
        "n_items": stats["total"],
        "tier1_coverage": {
            "exact": stats["match_exact"], "skeleton": stats["match_skeleton"],
            "no_pair": stats["match_no_pair"],
            "exact_pct": round(100 * stats["match_exact"] / n, 1),
            "no_pair_pct": round(100 * stats["match_no_pair"] / n, 1),
        },
        "no_pair_examples": no_pair_examples,
        "tier1_target_feature_agreement": {
            "agree": tf_agree[True], "disagree": tf_agree[False],
            "agree_pct": round(100 * tf_agree[True] / max(sum(tf_agree.values()), 1), 1),
        },
        "disagreement_examples": disagreement_examples,
        "zeyrek_available": _ANALYZER is not None,
        "zeyrek_init_error": _ZEYREK_INIT_ERROR,
        "zeyrek_parse_rate_pct": round(100 * stats["zeyrek_parsed"] / n, 1),
        "zeyrek_target_feature_agreement": {
            "agree": zy_tf_agree[True], "disagree": zy_tf_agree[False],
            "agree_pct": round(100 * zy_tf_agree[True] / max(sum(zy_tf_agree.values()), 1), 1),
        },
        "zeyrek_tier1_agreement": {
            "agree": zy_t1_agree[True], "disagree": zy_t1_agree[False],
            "agree_pct": round(100 * zy_t1_agree[True] / max(sum(zy_t1_agree.values()), 1), 1),
        },
    }
    return items, report


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", default=None,
                    choices=["train", "dev"], help="tekrarlanabilir; belirtilmezse ikisi de")
    ap.add_argument("--no-zeyrek", action="store_true")
    args = ap.parse_args()
    splits = args.split or ["train", "dev"]

    for split in splits:
        paths = sorted(DATA_DIR.glob(f"morph_{split}_v*.json"), reverse=True)
        if not paths:
            print(f"[atlandı] {split}: dosya yok")
            continue
        path = paths[0]
        d = json.loads(path.read_text(encoding="utf-8"))
        items, report = annotate_dataset(d["items"], use_zeyrek=not args.no_zeyrek)
        d["items"] = items
        d.setdefault("statistics", {})["morphology_annotation"] = report
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"=== {split} ({path.name}) ===")
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
