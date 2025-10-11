import json
from datetime import datetime, timedelta
from typing import List, Dict
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, 'memory.json')

class memory_using_system:
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

    def search_memory(key_word):
        return memory_string
