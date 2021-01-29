def test_clean_text():    
    from nltk import download
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    
    from main import process_description
    stemmer = SnowballStemmer("english")
    remove_words = set(stopwords.words("english"))

    before = "She is a lovely girl (the best)"
    after = ['love', 'girl', 'best']

    assert process_description(before,
                               stemmer,
                               remove_words) == after

    assert process_description("()[]{}@,;:/.?!§$#+-*=",
                               stemmer,
                               remove_words) == []


def test_read_data():
    from my_parser import local
    df_candidates = local.read()
    assert df_candidates.shape != (0, 0)
