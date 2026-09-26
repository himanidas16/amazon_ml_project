from business_er.normalize import _indic_info, has_indic


def test_fast_detector_matches_character_table():
    chars = [chr(i) for i in range(0x800,0xE00)]
    chars += [chr(i) for i in range(256)] + [chr(i) for i in [0x200c,0x200d,0xad,0xfeff,0x1f600]]
    for ch in chars:
        assert has_indic(ch) == (_indic_info(ch) is not None)
        assert has_indic('Business '+ch+' Road') == (_indic_info(ch) is not None)
    assert not has_indic('')
