import gradio as gr
import os
import time
from brain import TokenGPT
import pandas as pd
import base64

tk = TokenGPT()

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        with open(x, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
            img_html = f"<img src='data:image/png;base64,{b64_string}' />"
            history.append((img_html, None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)
'''
def bot(history):
    response = tk.conversation(history[-1][0])
    if response.endswith('.png'):
        path = f'/mnt/d/Code/tokenGPT/{response}'
        with open(path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
            img_html = f"<img src='data:image/png;base64,{b64_string}' />"
            history[-1][1] = img_html
    else:
        history[-1][1] = ""
        history[-1][1] = response
        # for character in response:
        #     history[-1][1] += character
        #     time.sleep(0.05)
        #     # yield history
    return history
'''

def bot(history):
    response = tk.conversation(history[-1][0])
    if isinstance(response, tuple):  # Check if the response is a tuple
        message,image_paths = response
        print(message)
        for img_path in image_paths:
            with open(img_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode()
                img_html = f"<img src='data:image/png;base64,{b64_string}' />"
                history.append((None, img_html))
        history.append((None, message))
    else:
        history[-1][1] = response
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

demo.queue()
demo.launch()
