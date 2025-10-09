import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "memory_mod_v1_2",
    "is_tab": False,
}

def output_modifier(string, state, is_chat=False):
    """
    關鍵字提取 搜尋記憶 組織回覆
    """
    return string