"""Text normalization: several "views" of every name and address.

The governing idea
------------------
Cleaning text is a BET.  Every rule says "these two strings mean the same
thing", and every rule can be wrong:

  * "St" -> "Street" is wrong when it meant "Saint".  Your data really contains
    `John Deere Street` matched against `John Deere Saint`.
  * Dropping "Ltd" helps when sources disagree about suffixes, and hurts if two
    different businesses differ only by suffix.
  * "12/3" -> "123" silently invents a different address.

Once a difference is erased, no later model can get it back.  So instead of
producing one cleaned string, we produce several VIEWS of each field and let the
model weigh them.  A conservative view keeps evidence; an aggressive view
bridges noise; disagreement between them is itself a signal.

The views
---------
  raw       exactly as supplied
  lower     Unicode-normalized and casefolded (S2 is mostly UPPERCASE, S3 mixed)
  plain     punctuation and repeated spaces collapsed
  folded    accents stripped -- undoes the injected `Empire` -> `Émpire` noise
  translit  Indic scripts phonetically romanized (see below)
  nosuffix  legal suffixes removed, kept SEPARATE so nothing is lost

Transliteration
---------------
About 23% of Indian Source-2 names are written in one of nine Indic scripts,
and they are phonetic spellings of the same Latin name -- `ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್`
is "Modern Consultants".  Measured on true pairs, name similarity for these
collapses to about 9.7 out of 100, so name matching simply does not work on them.

Unicode places these nine scripts in parallel 128-point blocks that share one
layout: offset 0x15 is "ka" in Devanagari, Bengali, Tamil, Kannada and the rest.
So a single table indexed by offset romanizes all nine.  This is local character
processing only -- no dictionary, no external data, no lookup service.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

NORMALIZE_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Indic transliteration
# --------------------------------------------------------------------------

# Start codepoint of each Indic block we handle.  They are 128 apart and share
# the Devanagari layout, which is what makes one table enough.
INDIC_BLOCKS = (
    0x0900,  # Devanagari  (Hindi, Marathi)
    0x0980,  # Bengali
    0x0A00,  # Gurmukhi    (Punjabi)
    0x0A80,  # Gujarati
    0x0B00,  # Oriya
    0x0B80,  # Tamil
    0x0C00,  # Telugu
    0x0C80,  # Kannada
    0x0D00,  # Malayalam
)

_CONS = "C"      # consonant: carries an inherent "a" unless cancelled
_VOWEL = "V"     # independent vowel
_MATRA = "M"     # dependent vowel sign: replaces the inherent "a"
_VIRAMA = "X"    # cancels the inherent "a"
_NASAL = "N"     # anusvara / candrabindu
_DIGIT = "D"
_SKIP = "-"      # nukta, ZWJ/ZWNJ and friends

# Offset within the block -> (kind, latin).  Consonants are stored WITHOUT the
# inherent vowel; it is added by the parser when nothing cancels it.
#
# Retroflex and dental consonants both map to plain "t"/"d", and the three "l"
# letters all map to "l".  That is deliberate: English spellings of Indian names
# vary freely between them, so collapsing helps matching rather than hurting it.
_OFFSETS: Dict[int, tuple] = {
    0x01: (_NASAL, "n"), 0x02: (_NASAL, "n"), 0x03: (_NASAL, "h"),
    # independent vowels
    0x05: (_VOWEL, "a"),  0x06: (_VOWEL, "a"), 0x07: (_VOWEL, "i"),
    0x08: (_VOWEL, "i"),  0x09: (_VOWEL, "u"),  0x0A: (_VOWEL, "u"),
    0x0B: (_VOWEL, "ri"), 0x0C: (_VOWEL, "li"), 0x0D: (_VOWEL, "e"),
    0x0E: (_VOWEL, "e"),  0x0F: (_VOWEL, "e"),  0x10: (_VOWEL, "ai"),
    0x11: (_VOWEL, "o"),  0x12: (_VOWEL, "o"),  0x13: (_VOWEL, "o"),
    0x14: (_VOWEL, "au"),
    # consonants
    0x15: (_CONS, "k"),   0x16: (_CONS, "kh"),  0x17: (_CONS, "g"),
    0x18: (_CONS, "gh"),  0x19: (_CONS, "ng"),  0x1A: (_CONS, "ch"),
    0x1B: (_CONS, "chh"), 0x1C: (_CONS, "j"),   0x1D: (_CONS, "jh"),
    0x1E: (_CONS, "ny"),  0x1F: (_CONS, "t"),   0x20: (_CONS, "th"),
    0x21: (_CONS, "d"),   0x22: (_CONS, "dh"),  0x23: (_CONS, "n"),
    0x24: (_CONS, "t"),   0x25: (_CONS, "th"),  0x26: (_CONS, "d"),
    0x27: (_CONS, "dh"),  0x28: (_CONS, "n"),   0x29: (_CONS, "n"),
    0x2A: (_CONS, "p"),   0x2B: (_CONS, "ph"),  0x2C: (_CONS, "b"),
    0x2D: (_CONS, "bh"),  0x2E: (_CONS, "m"),   0x2F: (_CONS, "y"),
    0x30: (_CONS, "r"),   0x31: (_CONS, "r"),   0x32: (_CONS, "l"),
    0x33: (_CONS, "l"),   0x34: (_CONS, "l"),   0x35: (_CONS, "v"),
    0x36: (_CONS, "sh"),  0x37: (_CONS, "sh"),  0x38: (_CONS, "s"),
    0x39: (_CONS, "h"),
    0x3C: (_SKIP, ""),                       # nukta
    # dependent vowel signs
    0x3E: (_MATRA, "a"), 0x3F: (_MATRA, "i"),  0x40: (_MATRA, "i"),
    0x41: (_MATRA, "u"),  0x42: (_MATRA, "u"),  0x43: (_MATRA, "ri"),
    0x44: (_MATRA, "ri"), 0x45: (_MATRA, "e"),  0x46: (_MATRA, "e"),
    0x47: (_MATRA, "e"),  0x48: (_MATRA, "ai"), 0x49: (_MATRA, "o"),
    0x4A: (_MATRA, "o"),  0x4B: (_MATRA, "o"),  0x4C: (_MATRA, "au"),
    0x4D: (_VIRAMA, ""),
    # additional consonants used for Perso-Arabic and English sounds
    0x58: (_CONS, "k"),   0x59: (_CONS, "kh"),  0x5A: (_CONS, "g"),
    0x5B: (_CONS, "j"),   0x5C: (_CONS, "r"),   0x5D: (_CONS, "r"),
    0x5E: (_CONS, "f"),   0x5F: (_CONS, "y"),
    0x60: (_VOWEL, "ri"), 0x61: (_VOWEL, "li"),
    0x62: (_MATRA, "l"),  0x63: (_MATRA, "l"),
    0x64: (_SKIP, " "),   0x65: (_SKIP, " "),   # danda punctuation
}
for _d in range(10):                             # Indic digits -> ASCII digits
    _OFFSETS[0x66 + _d] = (_DIGIT, str(_d))

_ZERO_WIDTH = {0x200C, 0x200D, 0x00AD}


def _indic_info(ch: str):
    """(kind, latin) for an Indic character, or None if it is not one."""
    cp = ord(ch)
    if cp in _ZERO_WIDTH:
        return (_SKIP, "")
    for start in INDIC_BLOCKS:
        if start <= cp < start + 0x80:
            return _OFFSETS.get(cp - start)
    return None


def has_indic(text: str) -> bool:
    return any(_indic_info(ch) is not None for ch in text)


def transliterate_indic(text: str, drop_final_a: bool = True) -> str:
    """Romanize any Indic text, leaving everything else untouched.

    The parser walks left to right.  A consonant emits its sound and then looks
    ahead: a following vowel sign supplies the vowel, a virama cancels it, and
    otherwise the inherent "a" is added.

    drop_final_a removes a word-final inherent "a", which is how Hindi is
    actually pronounced (schwa deletion).  It turns "limiteda" into "limited"
    and "besta" into "best", both much closer to the Latin spelling we are
    trying to match.  It is a flag so the gain can be measured rather than
    assumed.
    """
    if not text:
        return text

    # We build the output as alternating segments and remember which ones came
    # from Indic characters.  That matters for schwa deletion below: applying it
    # to Latin passthrough text would turn "area" into "are", and Indian
    # addresses are routinely mixed (Latin street, Devanagari state name).
    segments: List[List] = []          # [text, is_indic]

    def emit(chunk: str, indic: bool) -> None:
        if not chunk:
            return
        if segments and segments[-1][1] == indic:
            segments[-1][0] += chunk
        else:
            segments.append([chunk, indic])

    i, n = 0, len(text)
    while i < n:
        info = _indic_info(text[i])
        if info is None:
            emit(text[i], False)
            i += 1
            continue

        kind, latin = info
        if kind == _CONS:
            emit(latin, True)
            # Skip a nukta so it does not hide the vowel sign behind it.
            j = i + 1
            while j < n:
                nxt = _indic_info(text[j])
                if nxt and nxt[0] == _SKIP and nxt[1] == "":
                    j += 1
                else:
                    break
            nxt = _indic_info(text[j]) if j < n else None
            if nxt and nxt[0] == _VIRAMA:
                i = j + 1                      # bare consonant, no vowel
            elif nxt and nxt[0] == _MATRA:
                emit(nxt[1], True)
                i = j + 1
            else:
                emit("a", True)                # inherent vowel
                i = j
        elif kind in (_VOWEL, _MATRA, _NASAL, _DIGIT):
            emit(latin, True)
            i += 1
        else:                                   # virama or skip
            emit(latin, True)
            i += 1

    if drop_final_a:
        for seg in segments:
            if seg[1]:
                seg[0] = re.sub(r"(?<=[a-z]{2})a\b", "", seg[0])
    return "".join(seg[0] for seg in segments)


# --------------------------------------------------------------------------
# generic text views
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def to_lower(text: str) -> str:
    """Unicode-normalize then casefold.  NFKC also folds full-width and other
    compatibility forms onto their plain equivalents."""
    return unicodedata.normalize("NFKC", text).casefold()


def strip_accents(text: str) -> str:
    """Remove combining marks: `Émpire` -> `empire`, `Àmicale` -> `amicale`.

    The noise generator injects accents into both US and French names, so this
    view is worth having. It is applied AFTER transliteration so it never eats
    Indic vowel signs.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )


def collapse_punct(text: str) -> str:
    """Punctuation to spaces, then squeeze runs of whitespace.

    Punctuation becomes a SPACE rather than nothing, so `Private-Ltd` becomes
    two tokens instead of the single word `privateltd`.
    """
    return _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()


# Legal / organisational suffixes seen across all three countries.
# France matters here because it never appears in training: SARL, SAS, SASU,
# EURL, SCI and SA are all in the test set only.
LEGAL_SUFFIXES = frozenset("""
llc llp ltd limited inc incorporated corp corporation co company
pvt pvtltd private plc gmbh bv nv ag
sarl sas sasu eurl sci sa sca snc scop
and sons son bros brothers group holdings holding enterprises enterprise
services service solutions solution
""".split())

# Address abbreviations, applied token-wise (never as substring replacement, or
# "Stratford" would become "Streetratford").
#
# Note "st" is absent on purpose: it is ambiguous between Street and Saint, and
# the data contains `John Deere Street` matched to `John Deere Saint`.  We let
# character similarity handle it instead of guessing.
#
# This table is inevitably US/India-flavoured; it does NOT cover the French
# `R.`/`AV`/`BD` forms in the test set.  That is why the pipeline must lean on
# character-level features, which transfer, rather than on this dictionary.
ADDRESS_ABBREV = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building",
    "no": "number", "opp": "opposite", "nr": "near",
    "rue": "rue", "bd": "boulevard",
}


def drop_legal_suffixes(text: str) -> str:
    """Remove legal-suffix tokens, but never return an empty string.

    A business genuinely called "Services" would otherwise vanish entirely, and
    an empty name is far more dangerous than an unhelpful one: two empty strings
    look like a perfect match.
    """
    tokens = text.split()
    kept = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(kept) if kept else text


def expand_address_tokens(text: str) -> str:
    return " ".join(ADDRESS_ABBREV.get(t, t) for t in text.split())


def numbers_in(text: str) -> List[str]:
    """Digit groups, in order.  Street numbers, unit numbers and postal codes
    are among the strongest matching signals -- 80% of true pairs share at
    least one -- and they survive transliteration and language change."""
    return re.findall(r"\d+", text)


# --------------------------------------------------------------------------
# the view bundle
# --------------------------------------------------------------------------

VIEW_NAMES = ("lower", "plain", "folded", "translit", "nosuffix")


def name_views(raw: str) -> Dict[str, str]:
    """All views of a business name, cheapest transformation first."""
    lower = to_lower(raw)
    # Transliteration MUST come before collapse_punct: Indic vowel signs are
    # combining marks, which the punctuation regex would strip, leaving bare
    # consonants like "sh ra ba" instead of "shri best".
    translit = transliterate_indic(lower) if has_indic(lower) else lower
    plain = collapse_punct(translit)
    folded = strip_accents(plain)
    return {
        "raw": raw,
        "lower": lower,
        "translit": translit,
        "plain": plain,
        "folded": folded,
        "nosuffix": drop_legal_suffixes(folded),
    }


def address_views(raw: str) -> Dict[str, str]:
    """All views of an address, plus its digit groups."""
    lower = to_lower(raw)
    translit = transliterate_indic(lower) if has_indic(lower) else lower
    plain = collapse_punct(translit)
    folded = strip_accents(plain)
    return {
        "raw": raw,
        "lower": lower,
        "translit": translit,
        "plain": plain,
        "folded": folded,
        "expanded": expand_address_tokens(folded),
        "numbers": " ".join(numbers_in(raw)),
    }


def tokens(text: str) -> List[str]:
    return text.split()


def normalize_series(values: Iterable[str], kind: str = "name") -> List[Dict[str, str]]:
    fn = name_views if kind == "name" else address_views
    return [fn(v) for v in values]
