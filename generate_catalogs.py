import pandas as pd
import numpy as np
import random
import os

# Configuration
random.seed(42)
np.random.seed(42)
INPUT_CSV = "prepared_books_data.csv"
OUTPUT_DIR = "catalogs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_catalogs():
    print(f"Reading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Basic cleaning and preparation
    # Map lowercase 'title' to 'Title' if necessary, or just use what's there
    # The new file has: title,description,authors,publishedDate,categories,bookLength
    rename_map = {'title': 'Title', 'description': 'description', 'authors': 'authors', 
                  'publishedDate': 'publishedDate', 'categories': 'categories', 'bookLength': 'pages_count'}
    df = df.rename(columns=rename_map)
    
    # Generate a base price since prepared_books_data.csv doesn't have one
    print("Generating base prices...")
    df['base_price'] = np.round(np.random.uniform(10.0, 150.0, size=len(df)), 2)
    
    df = df.dropna(subset=['Title'])
    
    # --- Balanced Store Strategy (2 Specific, 1 Mega) ---

    # Store 1: Fiction Boutique (Cheaper)
    print("Generating 'fiction_boutique'...")
    fiction_mask = df['categories'].str.contains('Fiction|Science Fiction|Fantasy|Comics|Poetry|Juvenile Fiction', case=False, na=False)
    n_fiction = min(5000, fiction_mask.sum())
    store_1 = df[fiction_mask].sample(n=n_fiction, random_state=42).copy()
    store_1['price'] = np.round(store_1['base_price'] * 0.90, 2)
    store_1['shop_id'] = 'fiction_boutique'
    # 80% In Stock
    store_1['stock'] = np.random.choice([True, False], size=len(store_1), p=[0.8, 0.2])
    
    # Drop base_price before saving
    store_1 = store_1.drop(columns=['base_price'])
    store_1.to_csv(os.path.join(OUTPUT_DIR, "fiction_boutique.csv"), index=False)
    print(f"  - Created fiction_boutique.csv with {len(store_1)} items.")

    # Store 2: Knowledge Store (Expensive)
    print("Generating 'knowledge_store'...")
    knowledge_mask = df['categories'].str.contains('History|Biography|Religion|Social Science|Philosophy|Science|Technology', case=False, na=False)
    n_knowledge = min(5000, knowledge_mask.sum())
    store_2 = df[knowledge_mask].sample(n=n_knowledge, random_state=43).copy()
    store_2['price'] = np.round(store_2['base_price'] * 1.10, 2)
    store_2['shop_id'] = 'knowledge_store'
    # 80% In Stock
    store_2['stock'] = np.random.choice([True, False], size=len(store_2), p=[0.8, 0.2])
    
    store_2 = store_2.drop(columns=['base_price'])
    store_2.to_csv(os.path.join(OUTPUT_DIR, "knowledge_store.csv"), index=False)
    print(f"  - Created knowledge_store.csv with {len(store_2)} items.")

    # Store 3: Mega Market (Large Catalog for comparison)
    print("Generating 'mega_market'...")
    # Include ALL books from the dataset
    store_3 = df.copy()
    store_3['price'] = np.round(store_3['base_price'] * 1.0, 2)
    store_3['shop_id'] = 'mega_market'
    # 80% In Stock
    store_3['stock'] = np.random.choice([True, False], size=len(store_3), p=[0.8, 0.2])
    
    store_3 = store_3.drop(columns=['base_price'])
    store_3.to_csv(os.path.join(OUTPUT_DIR, "mega_market.csv"), index=False)
    print(f"  - Created mega_market.csv with {len(store_3)} items.")

    print("Done.")


def split_mega_market():
    df = pd.read_csv("catalogs/mega_market.csv")

    parts = 5
    chunk_size = len(df) // parts

    for i in range(parts):
        start = i * chunk_size
        end = None if i == parts - 1 else (i + 1) * chunk_size
        df.iloc[start:end].to_csv(f"mega_market{i + 1}.csv", index=False)


if __name__ == "__main__":
    #generate_catalogs()
    split_mega_market()
