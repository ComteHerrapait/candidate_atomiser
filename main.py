import re
from collections import Counter
from time import time

import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC

from my_parser import distant, local


def main():
    # > IMPORT data
    df_candidates = local.read()  # information about candidates : id, description, gender
    df_labels, df_categories = initialization()

    # > REDUCE size of dataset for faster tests
    # REMOVE BEFORE PRODUCTION
    df_labels = df_labels.iloc[:20000]
    df_candidates = df_candidates.iloc[:20000]

    print("\n\n>> Running process with {} samples :".format(df_candidates.shape[0]))
    # > PREPROCESS : clean description for easier analysis
    print(">> Processing descriptions ... ", end="\r", flush=True)
    time_start = time()

    # print("\n{}\n".format(df_candidates.iloc[42]["description"]))

    stemmer = SnowballStemmer("english")
    try:
        remove_words = set(stopwords.words("english"))
    except Exception as e:
        print("ERROR stopwords not found")
        print("FIX : attempting to download now ...")
        download('stopwords')  # download the stopwords file
        remove_words = set(stopwords.words("english"))

    for i, row in df_candidates.iterrows():
        df_candidates.at[i, 'bagOfWords'] = process_description(
            row["description"],
            stemmer,
            remove_words
        )

    # print("\n{}\n".format(df_candidates.iloc[42]["description"]))

    print(">> Processing descriptions done in : {:.2f}s".format(
        time()-time_start))
    # at this point descriptions are in bag of words form

    # > VECTORISE text : transform description into vectors to train the model
    vectorizer = CountVectorizer()
    term_matrix = vectorizer.fit_transform(df_candidates['bagOfWords'])
    print(">> Vectorization done :")
    print("\tfeatures count : {}".format(len(vectorizer.get_feature_names())))

    # > SPLIT data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        term_matrix, df_labels['Category'], test_size=0.33, random_state=0)
    # TODO : the training part of the data should be saved in a file name predict.csv
    print(">> dataset divided into {}|{} for test|train".format(
        X_test.shape[0], X_train.shape[0]))

    # TODO : training
    # > CHOOSE a model for training (SVC | )
    time_start = time()
    print("Training Model ...")
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(">> Model Trained in : {:.2f}s".format(
        time()-time_start))
    print(">> Accuracy = {:.2f}%".format(accuracy*100))

    print(model.n_support_)


def initialization():
    # annotation : associating each candidate with the correct job category
    df_labels = pd.read_csv("resources/label.csv")
    # names of the different job categories
    df_categories = pd.read_csv("resources/categories_string.csv")

    return df_labels, df_categories


def process_description(content_, stemmer=None, remove_words=[]):
    # remove capital characters
    content = content_.lower()

    # remove ponctuation and special characters, keep only letters
    content = " ".join(re.findall('[a-zA-Z]+', content))

    # replace words by their stem, splitting also remove double spaces
    words = content.split()
    words_stem = []
    if stemmer != None:
        for word in words:
            stem = stemmer.stem(word)
            if not stem in remove_words:
                words_stem.append(stem)
        return " ".join(words_stem)
    else:
        return " ".join(words)


def find_most_common_words(number=None, display=False):
    # read data from file
    df_candidates = local.read()
    #df_candidates = df_candidates.iloc[:50]
    if display:
        print("#> Processing most common words ...")
        time_start = time()

    # clean descriptions
    stemmer = SnowballStemmer("english")
    remove_words = set(stopwords.words("english"))
    for i, row in df_candidates.iterrows():
        df_candidates.at[i, 'description'] = process_description(
            row["description"],
            stemmer,
            remove_words
        )

    # count word occurences
    common_words = Counter()
    for i, row in df_candidates.iterrows():
        for word in row["description"]:
            common_words[word] += 1

    # display if wanted
    if display:
        print("### done {:.2f}s".format(time()-time_start))
        print(">|< Most common words >|<")
        top_words = common_words.most_common(number)
        for x in top_words:
            print("> {} \t: {} ".format(x[1], x[0]))  # beautifull display
            # print("\"{}\",".format(x[0]), end="") #ready for copy and paste

    return dict(common_words)


if __name__ == '__main__':
    #find_most_common_words(25, True)
    main()
