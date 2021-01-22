def test_clean_text():
    from main import clean_description
    before = "She is a lovely girl (the best)"
    after = "is lovely girl best"

    assert clean_description(before) == after

    assert clean_description("()[]{}@,;:/.?!§$#+-*=") == ""


def test_read_data():
    from my_parser import local
    df_candidates = local.read()
    assert df_candidates.shape != (0, 0)
