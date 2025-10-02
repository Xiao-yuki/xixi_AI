import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

from datetime import datetime

params = {
    "display_name": "time_mod",
    "is_tab": False,
}

def chat_input_modifier(text, visible_text, state):
    week_map = {
        "Monday":"星期一",
        "Tuesday":"星期二",
        "wednesday":"星期三",
        "Thursday":"星期四",
        "Friday":"星期五",
        "Saturday":"星期六",
        "Sunday":"星期日"
    }
    week_en = datetime.now().strftime("%A")
    now_1 = datetime.now().strftime("現在時間是 %Y年%m月%d日")
    now_2 = week_map[week_en]
    now_3 = datetime.now().strftime("%H:%M。")
    time_notice = f"[系統提示：時間資訊] {now_1} {now_2} {now_3}\n"

    modified_text = f"{time_notice}{text}"
    modified_visible = f"{time_notice}{visible_text}"

    return modified_text, modified_visible