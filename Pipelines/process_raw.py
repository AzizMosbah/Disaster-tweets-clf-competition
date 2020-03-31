import pandas as pd


def process(path: str) -> pd.DataFrame():
    """
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    df['text_original'] = df['text']
    return df

def replace_keyword_nans(df: pd.DataFrame()) -> pd.DataFrame():
    """
    :param df:
    :return:
    """
    df['keyword'] = df['keyword'].fillna("")
    df['location'] = df['location'].fillna("")

    return df
