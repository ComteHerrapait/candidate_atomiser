import re
from time import time

from collections import Counter
import pandas as pd
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
    df_labels = df_labels.iloc[:50]
    df_candidates = df_candidates.iloc[:50]

    # > PREPROCESS : clean description for easier analysis
    print(">> Processing descriptions ... ", end="\r", flush=True)
    time_start = time()
    for i, row in df_candidates.iterrows():
        df_candidates.at[i, 'description'] = clean_description(
            row["description"])
    print(">> Processing descriptions done in : {:.2f}".format(
        time()-time_start))

    # > SPLIT data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_candidates, df_labels, test_size=0.33, random_state=0)
    # TODO : the training part of the data should be saved in a file name predict.csv
    print(">> dataset divided into {}|{} for test|train".format(
        len(X_test), len(X_train)))

    # > VECTORISE text : transform description into vectors to train the model
    # TODO :
    # > CHOOSE a model for training (SVC | )
    # model = SVC(kernel="linear")
    # model.fit(X_train, y_train)
    # accuracy = model.score(X_test, y_test)
    # print("SVC model trained.")
    # print(">> Accuracy =", accuracy)

    # print("Class 0:", model.n_support_[0], "support vectors")
    # print("Class 1:", model.n_support_[1], "support vectors")


def initialization():
    # annotation : associating each candidate with the correct job category
    df_labels = pd.read_csv("resources/label.csv")
    # names of the different job categories
    df_categories = pd.read_csv("resources/categories_string.csv")

    return df_labels, df_categories


def clean_description(content):
    remove_word = ["and", "the", "of", "in", "a", "he", "to", "she", "is", "for", "has", "at", "his", "with", "on", "her",
                   "university", "as", "from", "s", "an", "dr", "was", "also", "years", "been", "school", "more", "that",
                   "this", "work", "by", "new", "received", "including", "center", "degree", "book", "currently", "are",
                   "science", "college", "other", "d", "or", "practice", "well", "be", "their", "than", "many", "have", "it",
                   "one", "over"]
    content = content.lower()
    content = " ".join(re.findall('[\w]+', content))
    content = " " + content + " "
    for word in remove_word:
        content = content.replace(" {} ".format(word), " ")
    content = " ".join(content.split())
    return content.strip()


def find_most_common_words(number=None, display=False):
    # read data from file
    df_candidates = local.read()
    if display:
        print("#> Processing most common words ...")
        time_start = time()

    # clean descriptions
    for i, row in df_candidates.iterrows():
        df_candidates.at[i, 'description'] = clean_description(
            row["description"])

    # count word occurences
    common_words = Counter()
    for i, row in df_candidates.iterrows():
        for word in row["description"].split():
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
    main()
