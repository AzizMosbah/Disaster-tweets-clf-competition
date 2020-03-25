from Pipelines.config import path
from Pipelines.process_raw import process, replace_keyword_nans
from Pipelines.transformations import tf_idf_table


def tfidf_pipeline(file: str, p: int = 600):
    processed_df = (
        process(file)
        .pipe(replace_keyword_nans)
        .pipe(tf_idf_table, feature_number= p, words='english')
    )
    return processed_df


if __name__ == "__main__":
    tfidf_pipeline(path)
