import pandas as pd

def load_and_prepare_data():
    df = pd.read_csv("dataset/fake_job_postings.csv")

    # Fill missing values
    text_columns = ["title", "description", "requirements", "company_profile", "benefits"]

    for col in text_columns:
        df[col] = df[col].fillna("")

    # Combine columns
    df["job_content"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["company_profile"] + " " +
        df["benefits"]
    )

    df = df[["job_content", "fraudulent"]]
    df = df.rename(columns={"fraudulent": "label"})

    return df