import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_FILE = os.path.join(BASE_DIR, 'words.xlsx')
JSON_FILE = os.path.join(BASE_DIR, 'words.json')

df = pd.read_excel(WORDS_FILE)
words = df.iloc[:, 0].dropna().tolist()
result = {word: i for i, word in enumerate(words)}

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(len(result))