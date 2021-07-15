import re
import pandas as pd
from functools import partial
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Uncomment when running first time
# import nltk
# nltk.download('stopwords')


# Function to clean a single sentence
def clean_sentence(sentence, return_sentence=True, remove_stopwords=True):
    sentence_no_letters = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence_lower_case = sentence_no_letters.lower()
    words = sentence_lower_case.split()

    # Remove stopwords
    if remove_stopwords:
        udf_stopwords = set(stopwords.words('english'))
    else:
        udf_stopwords = None

    no_stopwords = [word for word in words if not udf_stopwords or word not in udf_stopwords]

    # Merge words list to sentence
    if return_sentence:
        return ' '.join(no_stopwords)
    else:
        return no_stopwords


# Function to clean and load data
def load_data():
    data = pd.read_csv('./datasets/sentiment_analysis.csv')

    # Read and process data
    review_list = data.text.to_list()
    class_list = data.classes.to_list()
    sentiment_list = [1 if review == "Pos" else 0 for review in class_list]

    review_list = map(partial(clean_sentence, remove_stopwords=True), review_list)

    # Use count vectorizer to detect presence of word for bag of words implementation
    count_vect = CountVectorizer(binary=True)
    review_list_count = count_vect.fit_transform(review_list)
    review_list_count = review_list_count.toarray()

    # Train Test split
    X_train, X_test, Y_train, Y_test = train_test_split(review_list_count, sentiment_list, test_size=0.1,
                                                        random_state=99)

    return X_train, Y_train, X_test, Y_test
