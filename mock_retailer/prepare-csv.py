import pandas as pd
import numpy as np

def prepare_books_data_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv, dtype=str, na_values=["n/a", "NA", "N/A", "None", ""], keep_default_na=True)

    df = df[["Title", "description", "authors", "publishedDate", "categories"]]
    df = df.dropna()
    df = df.rename(columns={"Title": "title"})
    df["authors"] = df["authors"].str.strip("[]").str.replace("'", "", regex=False)
    df["categories"] = df["categories"].str.strip("[]").str.replace("'", "", regex=False)
    df = df.dropna()
    df = df[df["title"].ne("Just Like Mommy")]
    df.to_csv(output_csv, index=False)


def prepare_books_rating_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    df = df[["User_id", "Title", "review/summary", "review/score"]]
    df = df.dropna()
    df = df.rename(columns={"Title": "title", "review/summary": "review_summary",
                            "review/score": "review_score"})
    df.to_csv(output_csv, index=False)


def find_duplicate_titles(csv_path):
    df = pd.read_csv(csv_path)

    if "title" not in df.columns:
        raise ValueError("Column 'title' not found in CSV")

    duplicates = df[df.duplicated(subset="title", keep=False)]

    if duplicates.empty:
        print("No duplicate titles found.")
    else:
        print(f"Found {duplicates['title'].nunique()} duplicate titles:")
        print(duplicates.sort_values("title"))

    return duplicates


def add_random_page_count_to_csv(input_csv, output_csv, col_name="bookLength", seed=42):
    df = pd.read_csv(input_csv)

    rng = np.random.default_rng(seed)
    n = len(df)

    pages = rng.triangular(left=120, mode=260, right=450, size=n).round().astype(int)

    p_short = 0.01  # 4%
    p_long  = 0.02  # 3%

    short_mask = rng.random(n) < p_short
    long_mask  = (~short_mask) & (rng.random(n) < p_long)

    pages[short_mask] = rng.integers(30, 81, size=short_mask.sum())     # 30–80
    pages[long_mask]  = rng.integers(700, 1001, size=long_mask.sum())   # 700–1000

    df[col_name] = pages
    df.to_csv(output_csv, index=False)


def avg_description_chars(csv_path: str):
    df = pd.read_csv(csv_path)

    desc = df["description"].dropna().astype(str)
    desc = desc[desc.str.strip() != ""]

    char_counts = desc.str.len()

    print("Total rows:", len(df))
    print("Descriptions counted:", len(char_counts))
    print("Average chars:", char_counts.mean())
    print("Median chars:", char_counts.median())
    print("Max chars:", char_counts.max())
    print("Min chars:", char_counts.min())


def split_csv_into_parts(input_csv: str, out_prefix: str):
    df = pd.read_csv(input_csv)

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["title", "User_id"])

    parts = np.array_split(df, 6)

    for i, part in enumerate(parts, start=1):
        part.to_csv(f"{out_prefix}_{i}.csv", index=False)


def skip_first_rows(input, output):
    df = pd.read_csv(input, skiprows=range(1, 14972))
    df.to_csv(output, index=False)

if __name__ == "__main__":
    prepare_books_data_csv("data/books_data.csv", "data/prepared_books_data.csv")
    # prepare_books_rating_csv("data/books_rating.csv", "data/prepared_books_rating.csv")
    # find_duplicate_titles("data/prepared_books_data.csv")
    add_random_page_count_to_csv("data/prepared_books_data.csv", "data/prepared_books_data.csv")
    #avg_description_chars("data/prepared_books_data.csv")
    # split_csv_into_parts("data/prepared_books_rating.csv", "data/prepared_books_rating")
    #skip_first_rows("data/prepared_books_rating_part3.csv", "data/prepared_books_rating_part3.csv")

