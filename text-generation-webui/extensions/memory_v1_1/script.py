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
    "display_name": "memory_mod_v1",
    "is_tab": False,
}

def output_modifier(string, state, is_chat=False):
    """
    擷取指令 執行指令 回傳砍掉指令的回覆
    """
    return string