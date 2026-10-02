import os
import pandas as pd

def load_verses(base_dir):
    data = {}
    if not os.path.exists(base_dir):
        return data
    for book in os.listdir(base_dir):
        book_path = os.path.join(base_dir, book)
        if not os.path.isdir(book_path):
            continue
        for chapter in os.listdir(book_path):
            if not chapter.endswith('.txt'):
                continue
            chapter_path = os.path.join(book_path, chapter)
            with open(chapter_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        verse_num, text = parts
                        key = f"{book}_{chapter}_{verse_num}"
                        data[key] = text
    return data

def main():
    print("Loading Merina (NT)...")
    merina_data = load_verses("./NT")
    
    print("Loading Betsileo (NT_Betsileo)...")
    betsileo_data = load_verses("./NT_Betsileo")
    
    aligned = []
    
    for key in merina_data:
        if key in betsileo_data:
            aligned.append({
                "merina": merina_data[key],
                "betsileo": betsileo_data[key]
            })
            
    df = pd.DataFrame(aligned)
    print(f"Found {len(df)} aligned verses.")
    
    # Save to CSV
    out_file = "parallel_dataset.csv"
    df.to_csv(out_file, index=False)
    print(f"Dataset saved to {out_file}")

if __name__ == "__main__":
    main()
