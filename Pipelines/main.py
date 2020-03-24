from Pipelines.config import path
from Pipelines.process_raw import process, replace_index, replace_keyword_nans


def run_pipeline(file: str):
    processed_df = (
        process(file).pipe(replace_index).pipe(replace_keyword_nans)
    )
    return df


if __name__ == "__main__":
    run_pipeline(path)
