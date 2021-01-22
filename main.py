import re
from time import time

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
    remove_word = ["she", "he", "the", "a", "an",
                   "this", "that", "and", "of", "at",
                   "with", "by", "for", "then", "in", "as",
                   "s", "but"]
    content = content.lower()
    content = " ".join(re.findall('[\w]+', content))
    content = " " + content + " "
    for word in remove_word:
        content = content.replace(" {} ".format(word), " ")
    content = " ".join(content.split())
    return content.strip()


if __name__ == '__main__':
    main()
