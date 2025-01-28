import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score


nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def tweets_cleaning(tweet):
    """
    Cleans and preprocesses a tweet by:
    - Lowercasing
    - Removing URLs, HTML tags, punctuation, and numbers

    Returns:
        str: Cleaned tweet
    """
    # make tweet lowercase
    tweet = tweet.lower()
    # removing tweet within brackets
    tweet = re.sub("\[.*?\]", "", tweet)
    # removing tweet within parentheses
    tweet = re.sub("\(.*?\)", "", tweet)
    # removing numbers
    tweet = re.sub("\w*\d\w*", "", tweet)
    # if there's more than 1 whitespace, then make it just 1
    tweet = re.sub("\s+", " ", tweet)
    # if there's a new line, then make it a whitespace
    tweet = re.sub("\n", " ", tweet)
    # removing any quotes
    tweet = re.sub('"+', "", tweet)
    # removing &amp;
    tweet = re.sub("(\&amp\;)", "", tweet)
    # removing any usernames
    tweet = re.sub("(@[^\s]+)", "", tweet)
    # removing any hashtags
    tweet = re.sub("(#[^\s]+)", "", tweet)
    # remove `rt` for retweet
    tweet = re.sub("(rt)", "", tweet)
    # string.punctuation is a string of all punctuation marks
    # so this gets rid of all punctuation
    tweet = re.sub("[%s]" % re.escape(string.punctuation), "", tweet)
    # getting rid of `httptco`
    tweet = re.sub("(httptco)", "", tweet)
    return tweet


# tokenizing and removing stop words
def process_tweet(tweet):
    """
    tokenize text in each column and remove stop words
    """
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(tweet)
    stopwords_removed = [
        token.lower() for token in tokens if token.lower() not in stop_words
    ]
    return stopwords_removed


# function to print all evaluation metrics
def evaluation(precision, recall, f1_score, f1_weighted):
    """prints out evaluation metrics for a model"""
    print("Testing Set Evaluation Metrics:")
    print("Precision: {:.4}".format(precision))
    print("Recall: {:.4}".format(recall))
    print("F1 Score: {:.4}".format(f1_score))
    print("Weighted F1 Score: {:.4}".format(f1_weighted))


# function to print training cross validation f1 stats
def train_cross_validation(model, X_train, y_train, metric, x):
    """prints cross-validation TRAINING metrics for a model"""
    scores = cross_val_score(model, X_train, y_train, scoring=metric, cv=x)
    print("Cross-Validation F1 Scores on Training Set:", scores)
    print("\nMin: ", round(scores.min(), 6))
    print("Max: ", round(scores.max(), 6))
    print("Mean: ", round(scores.mean(), 6))
    print("Range: ", round(scores.max() - scores.min(), 6))


# determing out whether the model overfit or underfit
def model_fit(cv_train_metric, test_metric):
    if cv_train_metric > test_metric:
        models_fit = "overfit"
    else:
        models_fit = "underfit"
    return models_fit


# tokenizing specifically for Doc2Vec
def tokenize_text(text):
    """tokenize and remove stop words with NLTK"""
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def vec_for_learning(model, tagged_docs):
    """final vector feature for classifier use"""
    sents = tagged_docs.values
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents]
    )
    return targets, regressors
