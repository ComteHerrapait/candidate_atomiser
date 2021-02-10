import re
from collections import Counter
from time import time

import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from joblib import dump, load

from my_parser import distant, local


def main(sample_count=0):
    # > IMPORT data
    df_candidates = local.read()  # information about candidates : id, description, gender
    df_labels, df_categories = initialization()

    # > REDUCE size of dataset for faster tests
    # REMOVE BEFORE PRODUCTION
    if sample_count >= 1:
        df_labels = df_labels.iloc[5:sample_count]
        df_candidates = df_candidates.iloc[5:sample_count]

    print("\n\n>> Running process with {} samples :".format(
        df_candidates.shape[0]))
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

    # > VECTORISE text : transform description into vectors to train the model
    vectorizer = CountVectorizer()
    term_matrix = vectorizer.fit_transform(df_candidates['bagOfWords'])
    print("DEBUG : ", type(df_candidates['bagOfWords']))
    print(">> Vectorization done :")
    print("\tfeatures count : {}".format(len(vectorizer.get_feature_names())))

    # > CONVERT the occurence data into FREQUENCY data with TFIDF
    time_start = time()
    print(">> Converting to frequency of words (TFIDF) ...")
    tf_transformer = TfidfTransformer(sublinear_tf=True).fit(term_matrix)
    term_matrix_frequency = tf_transformer.transform(term_matrix)
    print(">> Convertion done in : {:.2f}s".format(time()-time_start))

    # > SPLIT data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        term_matrix_frequency, df_labels['Category'], test_size=0.33, random_state=0)
    # TODO : the training part of the data should be saved in a file name predict.csv
    print(">> dataset divided into {}|{} for test|train".format(
        X_test.shape[0], X_train.shape[0]))

    # > CHOOSE a model for training (SVC | )
    time_start = time()
    print(">> Training Model ...")
    # linear | poly | rbf | sigmoid | precomputed
    model = SVC(kernel="sigmoid")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(">> Model Trained in : {:.2f}s".format(
        time()-time_start))
    print(">> Accuracy = {:.2f}%".format(accuracy*100))

    print(">> supports vectors : ")
    if (df_categories.shape[0] > len(model.n_support_)):
        print("insuficient data : use more samples")
    else:
        for i in range(df_categories.shape[0]):
            print("\t{:3d}\t{} \t: {}".format(
                i, df_categories.iloc[i][0], model.n_support_[i]))

    # > SAVE model to disk to avoid recalculating everytime
    dump(model, 'resources/model.result')
    dump(vectorizer, 'resources/vectorizer.result')
    print(">> saved model to disk")


def initialization():
    """
    Initialize the job categories and the labels for the dataset from resource files.

    Returns:
        tuple (dataFrames): label dataframe and categories dataframe.
    """
    # annotation : associating each candidate with the correct job category
    df_labels = pd.read_csv("resources/label.csv")
    # names of the different job categories
    df_categories = pd.read_csv("resources/categories_string.csv")

    return df_labels, df_categories


def process_description(content_, stemmer=None, remove_words=[]):
    """
    Cleans a string and transforms it into a bag of words, by stemming word, putting it all in lower caps, and removing unnecessary characters.

    Args:
        content_ (string): The text to clean
        stemmer (stemmer, optional): Stemmer used for the stemming part of the cleaning. Defaults to None.
        remove_words (list, optional): List of useless words to remove from the text. Defaults to [].

    Returns:
        string: Bag of words corresponding to the input text.
    """
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


def predict_category_from_description(description):
    time_start = time()
    print("\n>> trying to predict job corresponding to :\n {} \n".format(description))
    model = load('resources/model.result')
    
    description_bagofwords = process_description(
        description,
        SnowballStemmer("english"),
        set(stopwords.words("english"))
    )
    print(">> bag of words :\n", description_bagofwords)
    
    vectorizer = load('resources/vectorizer.result')
    description_vectorized = vectorizer.transform([description_bagofwords])
    
    tf_transformer = TfidfTransformer(sublinear_tf=True).fit(description_vectorized)
    description_frequency = tf_transformer.transform(description_vectorized)
    
    index_predicted = model.predict(description_frequency)
    _, df_categories = initialization()
    print("Prediction acquired in {:.2f}s".format(time()-time_start))
    return df_categories.iloc[index_predicted]



def find_most_common_words(number=None, display=False):
    """
    This function returns the mosts common words in the descriptions of all candidates.

    Args:
        number (int, optional): Number of common words wanted. Defaults to None.
        display (bool, optional): Controls if the function displays something. Defaults to False.

    Returns:
        dict: A dictionnary containing the most common words and their count.
    """
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
    #main(20000)
    prediction  = predict_category_from_description(
        "She is also a Ronald D. Asmus Policy Entrepreneur Fellow with the German Marshall Fund and is a Visiting Fellow at the Centre for International Studies (CIS) at the University of Oxford. This commentary first appeared at Sada, an online journal published by the Carnegie Endowment for International Peace."
        )
    print("\n>> Result : ",prediction.iloc[0][0])
    prediction  = predict_category_from_description(
        "He is a member of the AICPA and WICPA. Brent graduated from the University of Wisconsin, La Crosse, with a degree in accountancy. He has lived in southern Wisconsin his entire life, and currently resides in Pardeeville with his wife Sara and their three children. Brent and Sara have a great appreciation of music, art, and theater."
        )
    print("\n>> Result : ",prediction.iloc[0][0])
    prediction  = predict_category_from_description(
        "Dr. Aster has held teaching and research positions at Ben Gurion University, Haifa University, Hebrew University, Bar Ilan University and the University of Pennsylvania. He has also taught Jewish studies at high schools in the United Stated and Israel."
        )
    print("\n>> Result : ",prediction.iloc[0][0])
    prediction  = predict_category_from_description(
        "He runs a boutique design studio attending clients in the United States, Europe and Asia. His work explores the convergence of human arts and science to give shape to an ever evolving design practice. With a particular commitment towards design education, Arturo is permanently engaged with the international design and development communities and often travels the world to exchange ideas about design with other designers and developers. Prior to his current venture, Arturo worked for Microsoft in Redmond for 7 years driving design evangelism"
        )
    print("\n>> Result : ",prediction.iloc[0][0])
