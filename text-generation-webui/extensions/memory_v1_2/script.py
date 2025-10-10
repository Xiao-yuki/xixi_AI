import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

from keyword_search import keyword_search_system
kwss = keyword_search_system()
from mem_sys import memory_using_system
mus = memory_using_system()

params = {
    "display_name": "memory_mod_v1_2",
    "is_tab": False,
}

def chat_input_modifier(text, visible_text, state):
    #關鍵字提取
    key_word = kwss.keyword_search(text)
    if key_word == "none":#如果不需要記憶就直接跳出
        return text, visible_text
    else:
        #搜尋記憶
        mem_str = mus.search_memory(key_word)
        #組織回覆
        m_text = f"{text}\n[系統提示：相關記憶] {mem_str}"
        m_visible_text = m_text#測試用
        return m_text, m_visible_text