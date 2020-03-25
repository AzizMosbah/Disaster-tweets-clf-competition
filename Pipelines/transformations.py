import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split




def tf_idf_transformer(df: pd.DataFrame, tvec: TfidfVectorizer):
    df = pd.DataFrame(tvec.transform(df.text).todense())
    df.columns = tvec.get_feature_names()
    return df


def tf_idf_table(
    tweets: pd.DataFrame, feature_number: int, words: str = None,
):

    tweets = tweets.sample(frac=1, random_state=123)
    X = tweets.loc[:, tweets.columns[:-1]]
    y = tweets["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=int(0.2 * len(X)), random_state=42
    )
    tvec = TfidfVectorizer(
        stop_words=words, max_features=feature_number, analyzer="word"
    )

    tvec.fit(X_train.text)

    X_train, X_test, y_train, y_test = (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

    train_tf_idf = tf_idf_transformer(X_train, tvec)
    test_tf_idf = tf_idf_transformer(X_test, tvec)

    X_train, X_test = X_train.join(train_tf_idf), X_test.join(test_tf_idf)

    return X_train, X_test, y_train, y_test
