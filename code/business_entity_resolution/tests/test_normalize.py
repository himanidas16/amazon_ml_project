"""Tests for text normalization and Indic transliteration.

Two kinds of test here:
  * transliteration produces something close to the Latin spelling
  * normalization does NOT destroy evidence -- the failure mode that silently
    lowers your score by making different businesses look identical
"""

import pytest

from business_er.normalize import (
    ADDRESS_ABBREV,
    LEGAL_SUFFIXES,
    address_views,
    collapse_punct,
    drop_legal_suffixes,
    expand_address_tokens,
    has_indic,
    name_views,
    numbers_in,
    strip_accents,
    to_lower,
    transliterate_indic,
)


# --------------------------------------------------------------------------
# transliteration
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "indic, must_contain",
    [
        ("श्री बेस्ट बिजनेस प्राइवेट लिमिटेड", ["best", "limited"]),      # Devanagari
        ("स्टार कंस्ट्रक्शन", ["star"]),
        ("राम मार्केटिंग", ["ram", "marketing"]),
        ("গোল্ড প্রডিউসার স্টোর্স লিমিটেড", ["gold", "limited"]),         # Bengali
        ("ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್", ["madarn"]),                          # Kannada
        ("શક્તિ અર્બન પ્રોડક્ટ્સ", ["shakti", "arban"]),                  # Gujarati
        ("குளோபல் பிசினஸ்", ["kulopal"]),                               # Tamil
        ("పర్‌ఫెక్ట్ యునైటెడ్", ["yunaited"]),                            # Telugu
        ("ശക്തി ഇംപെക്സ്", ["shakti"]),                                  # Malayalam
        ("ଭିଜନ୍ ଟେକ୍ନୋଲୋଜିସ୍", ["teknolojis"]),                          # Oriya
    ],
)
def test_transliteration_lands_near_the_latin_spelling(indic, must_contain):
    out = transliterate_indic(indic.lower())
    for token in must_contain:
        assert token in out, f"{out!r} missing {token!r}"


def test_all_nine_scripts_are_recognised():
    """Unicode lays these blocks out in parallel, which is why one table works.
    If a block were missing, that language would silently stay unmatchable."""
    samples = ["क", "ক", "ਕ", "ક", "କ", "க", "క", "ಕ", "ക"]
    assert len(samples) == 9
    for ch in samples:
        assert has_indic(ch)
        assert transliterate_indic(ch) == "ka"


def test_indic_state_names_romanize_onto_their_latin_spelling():
    """Real win: Source 1 writes "Uttar Pradesh", Source 2 writes it in
    Devanagari.  After transliteration the address tokens line up."""
    assert "uttar pradesh" in transliterate_indic("उत्तर प्रदेश")
    assert transliterate_indic("ಕರ್ನಾಟಕ").startswith("karnatak")


def test_virama_suppresses_the_inherent_vowel():
    assert transliterate_indic("क") == "ka"        # bare consonant keeps "a"
    assert transliterate_indic("क्") == "k"        # virama removes it
    assert transliterate_indic("का") == "ka"       # matra supplies the vowel
    assert transliterate_indic("कि") == "ki"


def test_indic_digits_become_ascii():
    """Numbers are among the strongest address signals, so they must not be
    lost just because they were written in a different script."""
    assert transliterate_indic("१२३") == "123"
    assert transliterate_indic("৪৫৬") == "456"


def test_non_indic_text_passes_through_untouched():
    for text in ["Émpire Douglas LLC", "SCI Ptit Àmicale", "12/3-A MG Road",
                 "Boulangerie Crème Brûlée", ""]:
        assert transliterate_indic(text) == text
        assert has_indic(text) is False


def test_schwa_deletion_only_touches_transliterated_text():
    """THE bug this guards: Indian addresses mix Latin street names with an
    Indic state name.  Applying schwa deletion to the whole string turned
    "area" into "are".  Latin words must survive byte for byte."""
    assert transliterate_indic("area plaza data extra") == "area plaza data extra"
    out = transliterate_indic("406, manas nagar colony, lucknow, उत्तर प्रदेश")
    assert "manas nagar colony" in out      # Latin part intact
    assert "uttar pradesh" in out           # Indic part romanized
    # And it still fires on genuinely Indic text.
    assert transliterate_indic("लिमिटेड") == "limited"


def test_transliteration_is_deterministic():
    text = "श्री बेस्ट बिजनेस"
    assert transliterate_indic(text) == transliterate_indic(text)


# --------------------------------------------------------------------------
# the generic views
# --------------------------------------------------------------------------

def test_accents_are_folded():
    """The noise generator injects accents: Empire -> Émpire, Amicale -> Àmicale."""
    assert strip_accents("Émpire") == "Empire"
    assert strip_accents("Àmicale Crème Brûlée") == "Amicale Creme Brulee"


def test_punctuation_becomes_a_space_not_nothing():
    """`Private-Ltd` must become two tokens.  Deleting the hyphen instead would
    give `privateltd`, which matches nothing."""
    assert collapse_punct("private-ltd") == "private ltd"
    assert collapse_punct("Mathews, Maez & Autrey") == "Mathews Maez Autrey"
    assert collapse_punct("  a   b  ") == "a b"


def test_casefolding_handles_the_uppercase_source():
    """Source 2 is largely uppercase, Source 3 mixed."""
    assert to_lower("PH EMPIRE DOUGLAS LLC") == to_lower("PH Empire Douglas llc")


def test_legal_suffixes_cover_france_which_is_unseen_in_training():
    """France is 15% of the test set and never appears in training, so its
    suffixes must be in the list from the start."""
    for suffix in ("sarl", "sas", "sasu", "eurl", "sci", "sa"):
        assert suffix in LEGAL_SUFFIXES
    assert drop_legal_suffixes("thermal fils sasu") == "thermal fils"


def test_dropping_suffixes_never_returns_an_empty_string():
    """A business really called "Services" must not vanish.  Two empty strings
    would look like a perfect match, which is the worst possible outcome under a
    precision-weighted metric."""
    assert drop_legal_suffixes("services") == "services"
    assert drop_legal_suffixes("ltd") == "ltd"
    assert drop_legal_suffixes("") == ""


def test_address_abbreviations_apply_token_wise_not_as_substrings():
    """Substring replacement would turn "Stratford" into "Streetratford" and
    "Drive" into "Drivive"."""
    assert expand_address_tokens("12 mg rd") == "12 mg road"
    assert expand_address_tokens("stratford") == "stratford"
    assert expand_address_tokens("dr") == "drive"
    assert expand_address_tokens("drive") == "drive"


def test_ambiguous_st_is_deliberately_not_expanded():
    """"St" is Street or Saint, and this dataset really contains
    `John Deere Street` matched against `John Deere Saint`.  Guessing would
    destroy evidence, so we leave it to character similarity."""
    assert "st" not in ADDRESS_ABBREV
    assert expand_address_tokens("john deere st") == "john deere st"


def test_numbers_are_extracted_in_order_and_structure_is_not_invented():
    assert numbers_in("12/3-A MG Road, 560001") == ["12", "3", "560001"]
    # "12/3" must not silently become "123" -- that is a different address.
    assert numbers_in("12/3") != ["123"]


# --------------------------------------------------------------------------
# view bundles
# --------------------------------------------------------------------------

def test_name_views_expose_every_expected_view():
    v = name_views("PH Émpire Douglas LLC")
    assert set(v) == {"raw", "lower", "translit", "plain", "folded", "nosuffix"}
    assert v["raw"] == "PH Émpire Douglas LLC"       # original always kept
    assert v["folded"] == "ph empire douglas llc"
    assert v["nosuffix"] == "ph empire douglas"


def test_address_views_include_numbers_and_expansion():
    v = address_views("12/3, MG Rd, Bengaluru, 560001")
    assert v["numbers"] == "12 3 560001"
    assert "road" in v["expanded"]
    assert v["raw"] == "12/3, MG Rd, Bengaluru, 560001"


def test_views_handle_empty_and_whitespace_only_text():
    """Real records have blank addresses -- one French test record has none at
    all.  Nothing may crash, and nothing may invent content."""
    for bad in ("", "   ", "\t"):
        n, a = name_views(bad), address_views(bad)
        assert n["folded"] == ""
        assert a["folded"] == "" and a["numbers"] == ""


def test_cross_script_pair_becomes_comparable():
    """End-to-end proof of the point of this module: two spellings of the same
    business that shared almost nothing now share most of their text."""
    from rapidfuzz import fuzz
    a = name_views("Shree Best Business Private Limited")["folded"]
    b = name_views("श्री बेस्ट बिजनेस प्राइवेट लिमिटेड")["folded"]
    assert fuzz.ratio(a, b) > 70
    raw_score = fuzz.ratio(
        "shree best business private limited", "श्री बेस्ट बिजनेस प्राइवेट लिमिटेड"
    )
    assert raw_score < 20          # unmatchable before


# ---- country-scoped abbreviations -------------------------------------------

def test_french_abbreviations_apply_only_to_france():
    from business_er.normalize import address_views
    fr = address_views("63 R. DE DIEPPE, ST MALO", "France")["expanded"]
    assert "rue" in fr.split() and "saint" in fr.split()
    us = address_views("105 ELM ST, R A KIDWAI", "US")["expanded"]
    assert "saint" not in us.split() and "rue" not in us.split()      # Street / initial untouched
    unknown = address_views("63 R. DE DIEPPE", "Narnia")["expanded"]
    assert "rue" not in unknown.split()                               # open set: no table, no crash
    assert address_views("63 R. DE DIEPPE")["expanded"] == unknown    # default = no country table
