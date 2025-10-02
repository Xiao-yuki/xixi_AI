import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, 'memory.json')

SHORT_TERM_THRESHOLD = 0.8  # 升級長期記憶門檻
FORGET_THRESHOLD = 0.2      # 被遺忘的門檻
DECAY_DAYS = 3              # 幾天沒用會開始衰退

class MemorySystem:
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.memories: List[Dict] = []
        self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.memories = []

    def save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)

    def add_memory(self, content: str, keywords: List[str], preference: float = 0.0):
        mem = {
            "id": str(uuid.uuid4()),
            "content": content,
            "keywords": keywords,
            "frequency": 0.1,  # 初始頻率
            "preference": preference,
            "last_used": datetime.now().isoformat(),
            "type": "short_term"
        }
        self.memories.append(mem)
        self.save_memory()
        return mem

    def search_memory(self, query_keywords: List[str], top_k=5):
        now = datetime.now()
        results = []
        for mem in self.memories:
            match_score = len(set(query_keywords) & set(mem['keywords']))
            if match_score > 0:
                last_used_dt = datetime.fromisoformat(mem['last_used'])
                days_ago = (now - last_used_dt).days
                recency_weight = 1 / (1 + days_ago)  # 越近越高，避免除0用1+

                score = mem['frequency'] * (1 + mem['preference']) * recency_weight
                results.append((score, mem))
        results.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in results[:top_k]]

    def use_memory(self, memory_id: str):
        for mem in self.memories:
            if mem['id'] == memory_id:
                # 更新 frequency（喜好影響增長）
                increment = 0.1 * (1 + mem['preference'])
                mem['frequency'] = min(mem['frequency'] + increment, 1.0)
                mem['last_used'] = datetime.now().isoformat()

                # 檢查是否升級為長期記憶
                if mem['type'] == 'short_term' and mem['frequency'] >= SHORT_TERM_THRESHOLD:
                    mem['type'] = 'long_term'

                self.save_memory()
                return mem
        return None

    def decay_memory(self):
        now = datetime.now()
        updated = False
        for mem in self.memories[:]:
            last_used = datetime.fromisoformat(mem['last_used'])
            days_passed = (now - last_used).days

            if days_passed >= DECAY_DAYS:
                decay = 0.05 * days_passed * (1 - mem['preference'])  # 喜歡的衰退較慢
                mem['frequency'] -= decay
                mem['frequency'] = max(mem['frequency'], 0.0)
                updated = True

                # 自動忘記短期記憶
                if mem['type'] == 'short_term' and mem['frequency'] < FORGET_THRESHOLD:
                    self.memories.remove(mem)
                    updated = True

        if updated:
            self.save_memory()

    def print_memory(self):
        for mem in self.memories:
            print(f"[{mem['type'].upper()}] {mem['content']}")
            print(f"  ➤ Keywords: {mem['keywords']}")
            print(f"  ➤ Frequency: {mem['frequency']:.2f}, Preference: {mem['preference']}")
            print(f"  ➤ Last used: {mem['last_used']}")
            print("")
