import json
import gradio as gr
# import openai
import os
import sys
import traceback
import requests
# import markdown
import csv

my_api_key = ""    # ここに API キーを入力してください
initial_prompt = "You are a helpful assistant."
API_URL = "https://api.openai.com/v1/chat/completions"
HISTORY_DIR = "history"
TEMPLATES_DIR = "templates"



#if we are running in Docker
if os.environ.get('dockerrun') == 'yes':
    dockerflag = True
else:
    dockerflag = False

if dockerflag:
    my_api_key = os.environ.get('my_api_key')
    if my_api_key == "empty":
        print("Please give a api key!")
        sys.exit(1)
    #auth
    username = os.environ.get('USERNAME')
    password = os.environ.get('PASSWORD')
    if isinstance(username, type(None)) or isinstance(password, type(None)):
        authflag = False
    else:
        authflag = True


def parse_text(text):
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="{items[-1]}">'
            else:
                lines[i] = f'</code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("&", "&amp;")
                    line = line.replace("\"", "&quot;")
                    line = line.replace("\'", "&apos;")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                lines[i] = '<br/>'+line
    return "".join(lines)

def predict(inputs, top_p, temperature, openai_api_key, chatbot=[], history=[], system_prompt=initial_prompt, retry=False, summary=False):  # repetition_penalty, top_k

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    chat_counter = len(history) // 2

    print(f"chat_counter - {chat_counter}")

    messages = [compose_system(system_prompt)]
    if chat_counter:
        for data in chatbot:
            temp1 = {}
            temp1["role"] = "user"
            temp1["content"] = data[0]
            temp2 = {}
            temp2["role"] = "assistant"
            temp2["content"] = data[1]
            if temp1["content"] != "":
                messages.append(temp1)
                messages.append(temp2)
            else:
                messages[-1]['content'] = temp2['content']
    if retry and chat_counter:
        messages.pop()
    elif summary and chat_counter:
        messages.append(compose_user(
            "会話の質を確保しながら文字数を減らすために、上記の会話の内容を要約するのを手伝ってください。この文を要約に含めないでください。"))
        history = ["今何について話しましたか？"]
    else:
        temp3 = {}
        temp3["role"] = "user"
        temp3["content"] = inputs
        messages.append(temp3)
        chat_counter += 1
    # messages
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,  # [{"role": "user", "content": f"{inputs}"}],
        "temperature": temperature,  # 1.0,
        "top_p": top_p,  # 1.0,
        "n": 1,
        "stream": True,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    if not summary:
        history.append(inputs)
    print(f"payload is - {payload}")
    # make a POST request to the API endpoint using the requests.post method, passing in stream=True
    response = requests.post(API_URL, headers=headers,
                             json=payload, stream=True)
    #response = requests.post(API_URL, headers=headers, json=payload, stream=True)

    token_counter = 0
    partial_words = ""

    counter = 0
    chatbot.append((history[-1], ""))
    for chunk in response.iter_lines():
        if counter == 0:
            counter += 1
            continue
        counter += 1
        # check whether each line is non-empty
        if chunk:
            # decode each line as response data is in bytes
            try:
                if len(json.loads(chunk.decode()[6:])['choices'][0]["delta"]) == 0:
                    break
            except Exception as e:
                chatbot.pop()
                chatbot.append((history[-1], f"☹エラーが発生しました\n戻り値：{response.text}\エラー：{e}"))
                history.pop()
                yield chatbot, history
                break
            #print(json.loads(chunk.decode()[6:])['choices'][0]["delta"]    ["content"])
            partial_words = partial_words + \
                json.loads(chunk.decode()[6:])[
                    'choices'][0]["delta"]["content"]
            if token_counter == 0:
                history.append(" " + partial_words)
            else:
                history[-1] = parse_text(partial_words)
            chatbot[-1] = (history[-2], history[-1])
        #   chat = [(history[i], history[i + 1]) for i in range(0, len(history)     - 1, 2) ]  # convert to tuples of list
            token_counter += 1
            # resembles {chatbot: chat,     state: history}
            yield chatbot, history



def delete_last_conversation(chatbot, history):
    if chat_counter > 0:
        chat_counter -= 1
        chatbot.pop()
        history.pop()
        history.pop()
    return chatbot, history

def save_chat_history(filename, system, history, chatbot):
    if filename == "":
        return
    if not filename.endswith(".json"):
        filename += ".json"
    os.makedirs(HISTORY_DIR, exist_ok=True)
    json_s = {"system": system, "history": history, "chatbot": chatbot}
    with open(os.path.join(HISTORY_DIR, filename), "w") as f:
        json.dump(json_s, f)


def load_chat_history(filename):
    with open(os.path.join(HISTORY_DIR, filename), "r") as f:
        json_s = json.load(f)
    return filename, json_s["system"], json_s["history"], json_s["chatbot"]


def get_file_names(dir, plain=False, filetype=".json"):
    # find all json files in the current directory and return their names
    try:
        files = [f for f in os.listdir(dir) if f.endswith(filetype)]
    except FileNotFoundError:
        files = []
    if plain:
        return files
    else:
        return gr.Dropdown.update(choices=files)

def get_history_names(plain=False):
    return get_file_names(HISTORY_DIR, plain)

def load_template(filename):
    lines = []
    with open(os.path.join(TEMPLATES_DIR, filename), "r") as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)
    lines = lines[1:]
    return {row[0]:row[1] for row in lines}, gr.Dropdown.update(choices=[row[0] for row in lines])

def get_template_names(plain=False):
    return get_file_names(TEMPLATES_DIR, plain, filetype=".csv")

def reset_state():
    return [], []


def compose_system(system_prompt):
    return {"role": "system", "content": system_prompt}


def compose_user(user_input):
    return {"role": "user", "content": user_input}


def reset_textbox():
    return gr.update(value='')

title = """<h1 align="center">川虎ChatGPT 日本語版　🚀</h1>"""
description = """<div align=center>

由Bilibili [土川虎虎虎](https://space.bilibili.com/29125536) 开发

访问川虎ChatGPT的 [GitHub项目](https://github.com/GaiZhenbiao/ChuanhuChatGPT) スクリプトの最新バージョンをダウンロードする

このアプリは「gpt-3.5-turbo」ビッグ ランゲージ モデルを使用しています
</div>
"""
with gr.Blocks() as demo:
    gr.HTML(title)
    keyTxt = gr.Textbox(show_label=True, placeholder=f"ここにあなたのOpenAI API-keyを入力してください...",
                        value=my_api_key, label="API Key", type="password").style(container=True)
    chatbot = gr.Chatbot()  # .style(color_map=("#1D51EE", "#585A5B"))
    history = gr.State([])
    promptTemplates = gr.State({})
    TRUECOMSTANT = gr.State(True)
    FALSECONSTANT = gr.State(False)
    topic = gr.State("無題の会話履歴")

    with gr.Row():
        with gr.Column(scale=12):
            txt = gr.Textbox(show_label=False, placeholder="ここに入力").style(
                container=False)
        with gr.Column(min_width=50, scale=1):
            submitBtn = gr.Button("🚀", variant="primary")
    with gr.Row():
        emptyBtn = gr.Button("🧹 新しい対話")
        retryBtn = gr.Button("🔄 再生する")
        delLastBtn = gr.Button("🗑️ 最後の会話を削除する")
        reduceTokenBtn = gr.Button("♻️ 会話を要約する")
    systemPromptTxt = gr.Textbox(show_label=True, placeholder=f"ここにSystem Promptを入力...",
                                 label="System prompt", value=initial_prompt).style(container=True)
    with gr.Accordion(label="Prompt Templateをロードする", open=False):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=6):
                    templateFileSelectDropdown = gr.Dropdown(label="プロンプト テンプレート コレクション ファイルを選択します（.csv）", choices=get_template_names(plain=True), multiselect=False)
                with gr.Column(scale=1):
                    templateRefreshBtn = gr.Button("🔄 更新")
                    templaeFileReadBtn = gr.Button("📂 テンプレートを読み込む")
            with gr.Row():
                with gr.Column(scale=6):
                    templateSelectDropdown = gr.Dropdown(label="プロンプト テンプレートから読み込む", choices=[], multiselect=False)
                with gr.Column(scale=1):
                    templateApplyBtn = gr.Button("⬇️ 応用")
    with gr.Accordion(label="会話履歴の保存/読み込み (テキスト ボックスにファイル名を入力し、[会話を保存] ボタンをクリックすると、Python ファイルの隣に履歴ファイルが保存されます)", open=False):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=6):
                    saveFileName = gr.Textbox(
                        show_label=True, placeholder=f"保存したファイル名をここに入力...", label="保存ファイル名を設定", value="会話履歴").style(container=True)
                with gr.Column(scale=1):
                    saveBtn = gr.Button("💾 会話を保存")
            with gr.Row():
                with gr.Column(scale=6):
                    historyFileSelectDropdown = gr.Dropdown(label="リストから会話を読み込む", choices=get_history_names(plain=True), multiselect=False)
                with gr.Column(scale=1):
                    historyRefreshBtn = gr.Button("🔄 更新")
                    historyReadBtn = gr.Button("📂 会話で読む")
    #inputs, top_p, temperature, top_k, repetition_penalty
    with gr.Accordion("パラメータ", open=False):
        top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.05,
                          interactive=True, label="Top-p (nucleus sampling)",)
        temperature = gr.Slider(minimum=-0, maximum=5.0, value=1.0,
                                step=0.1, interactive=True, label="Temperature",)
        #top_k = gr.Slider( minimum=1, maximum=50, value=4, step=1, interactive=True, label="Top-k",)
        #repetition_penalty = gr.Slider( minimum=0.1, maximum=3.0, value=1.03, step=0.01, interactive=True, label="Repetition Penalty", )
    gr.Markdown(description)


    txt.submit(predict, [txt, top_p, temperature, keyTxt,
               chatbot, history, systemPromptTxt], [chatbot, history])
    txt.submit(reset_textbox, [], [txt])
    submitBtn.click(predict, [txt, top_p, temperature, keyTxt, chatbot,
                    history, systemPromptTxt], [chatbot, history], show_progress=True)
    submitBtn.click(reset_textbox, [], [txt])
    emptyBtn.click(reset_state, outputs=[chatbot, history])
    retryBtn.click(predict, [txt, top_p, temperature, keyTxt, chatbot, history,
                   systemPromptTxt, TRUECOMSTANT], [chatbot, history], show_progress=True)
    delLastBtn.click(delete_last_conversation, [chatbot, history], [
                     chatbot, history], show_progress=True)
    reduceTokenBtn.click(predict, [txt, top_p, temperature, keyTxt, chatbot, history,
                         systemPromptTxt, FALSECONSTANT, TRUECOMSTANT], [chatbot, history], show_progress=True)
    saveBtn.click(save_chat_history, [
                  saveFileName, systemPromptTxt, history, chatbot], None, show_progress=True)
    saveBtn.click(get_history_names, None, [historyFileSelectDropdown])
    historyRefreshBtn.click(get_history_names, None, [historyFileSelectDropdown])
    historyReadBtn.click(load_chat_history, [historyFileSelectDropdown],  [saveFileName, systemPromptTxt, history, chatbot], show_progress=True)
    templateRefreshBtn.click(get_template_names, None, [templateFileSelectDropdown])
    templaeFileReadBtn.click(load_template, [templateFileSelectDropdown],  [promptTemplates, templateSelectDropdown], show_progress=True)
    templateApplyBtn.click(lambda x, y: x[y], [promptTemplates, templateSelectDropdown],  [systemPromptTxt], show_progress=True)

print("Chuanhu からのヒント: http://localhost:7860 にアクセスしてインターフェイスを表示します")
# 默认开启本地服务器，默认可以直接从IP访问，默认不创建公开分享链接
demo.title = "川虎ChatGPT　日本語版 🚀"

#if running in Docker
if dockerflag:
    if authflag:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860,auth=(username, password))
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
#if not running in Docker
else:
    demo.queue().launch(server_name = "0.0.0.0", share=False) # 改为 share=True 可以创建公开分享链接
    #demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False) # 可自定义端口
    #demo.queue().launch(server_name="0.0.0.0", server_port=7860,auth=("在这里填写用户名", "在这里填写密码")) # 可设置用户名与密码
