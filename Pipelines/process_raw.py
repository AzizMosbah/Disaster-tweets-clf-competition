import pandas as pd


def process(path: str) -> pd.DataFrame():
    """
    :param path:
    :return:
    """
    return pd.read_csv(path)


def replace_index(df: pd.DataFrame()) -> pd.DataFrame():

    """
    :param df:
    :return:
    """

    df.index = df["id"]
    df = df.drop(columns=["id"], axis=1)
    return df


def replace_keyword_nans(df: pd.DataFrame()) -> pd.DataFrame():

    """

    :param df:
    :return:
    """
    df = df["keyword"].fillna("")
    return df
