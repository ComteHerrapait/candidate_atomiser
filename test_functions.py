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


def test_prediction():
    from main import predict_category_from_description, initialization
    from my_parser import local
    from random import randint
    df_candidates = local.read()
    df_labels, df_categories = initialization()
    random_choice = randint(0, df_candidates.shape[0])
    prediction = predict_category_from_description(
        df_candidates.iloc[random_choice]['description']
    )
    assert len(prediction) > 0
