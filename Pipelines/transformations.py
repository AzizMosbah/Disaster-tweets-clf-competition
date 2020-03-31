import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    CountVectorizer,
)
from sklearn.model_selection import train_test_split

from nltk.stem.snowball import SnowballStemmer

# Because the data frame is very small I input it as a whole instead of inputting just the series column
# For production code this might be quite important as not only does it ease testing but also best for memory
# This is an analysis code so it has to be pleasing to the eyes as well.


def remove_urls(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    df.text = df.text.str.replace("http\S+|www.\S+", "", case=False)
    return df


def remove_emojis(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    emojis = [
        u"\U0001F600-\U0001F64F",
        u"\U0001F300-\U0001F5FF",
        u"\U0001F680-\U0001F6FF",
        u"\U0001F1E0-\U0001F1FF",
        u"\U00002702-\U000027B0",
        u"\U000024C2-\U0001F251",
    ]
    df.text = df.text.str.replace("|".join(emojis), "", case=False,)
    return df


def remove_punctuations(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    df.text = df.text.str.replace("[^\w\s]", "")
    return df


def nltk_stemmer(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    stemmer = SnowballStemmer("english")
    df.text = df.text.str.split()
    df.text = df.text.apply(lambda x: [stemmer.stem(y) for y in x])
    df.text = df.text.apply(lambda x: " ".join(x))
    return df


def vector_transformer(df: pd.DataFrame, vec):
    """

    :param df:
    :param tvec:
    :return:
    """
    df = pd.DataFrame(vec.transform(df.text).todense())
    df.columns = vec.get_feature_names()
    return df


def tf_idf_table(
    tweets: pd.DataFrame, feature_number: int, words: str = None, vec: str = "tfidf"
):
    """

    :param tweets:
    :param feature_number:
    :param words:
    :param vec:
    :return:
    """
    tweets = tweets.sample(frac=1, random_state=123)
    y = tweets["target"]
    X = tweets.loc[:, tweets.columns != "target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=int(0.2 * len(X)), random_state=42
    )
    if vec == "tfidf":
        tvec = TfidfVectorizer(
            stop_words=words, max_features=feature_number, analyzer="word"
        )
    elif vec == "count":
        tvec = CountVectorizer(
            stop_words=words, max_features=feature_number, analyzer="word"
        )
    elif vec == "hash":
        tvec = HashingVectorizer(
            stop_words=words, max_features=feature_number, analyzer="word"
        )

    tvec.fit(X_train.text)

    X_train, X_test, y_train, y_test = (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

    train_tf_idf = vector_transformer(X_train, tvec)
    train_tf_idf = train_tf_idf.rename(columns={"id": "ids"})
    test_tf_idf = vector_transformer(X_test, tvec)
    test_tf_idf = test_tf_idf.rename(columns={"id": "ids"})
    X_train, X_test = X_train.join(train_tf_idf), X_test.join(test_tf_idf)

    return X_train, X_test, y_train, y_test
