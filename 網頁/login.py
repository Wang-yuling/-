# login.py
import pandas as pd
import logging
import os
import base64
from datetime import datetime, timezone, timedelta
from flask_socketio import SocketIO, emit
from decouple import config
from openai import AzureOpenAI
from flask import Flask, request, render_template, redirect, url_for, flash
import speech_recognition as sr
from threading import Thread
from werkzeug.utils import secure_filename
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


UPLOAD_FOLDER = 'uploads'  # 上傳到 uploads 資料夾
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

result_history = []
valid_credentials_s = {}
valid_credentials_t = {}
photos_folder_path = 'uploads'  # 路徑

subscription_key = 'b09eb5d6ad4e4395bf940237b2e5de8d'
endpoint = 'https://imageprocessinggroup.cognitiveservices.azure.com/'
credentials = CognitiveServicesCredentials(subscription_key)
# 勿更動位置 ######################################################
client = ComputerVisionClient(endpoint, credentials)


logging.basicConfig(level=logging.INFO)  # INFO可能之後可以改其他logging level
logger = logging.getLogger(__name__)  # __name__不確定樣不要改成mudule名


# 初始化 AzureOpenAI 客戶端
azure_endpoint = config("AZURE_OPENAI_ENDPOINT",  # 不要動
                        default="https://grade3projectchat.openai.azure.com/")  # 不要動
api_version = config("AZURE_OPENAI_API_VERSION",
                     default="2023-07-01-preview")  # 不要動
api_key = config("AZURE_OPENAI_API_KEY",  # 不要動
                 default="939c89a02f244fb19e32caeb1f85d0d6")  # 不要動
client = AzureOpenAI(  # 不要動
    azure_endpoint=azure_endpoint,  # 不要動
    api_version=api_version,  # 不要動
    api_key=api_key  # 不要動
)

# 沒有資料夾自動新增
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file has one of the allowed extensions."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# 上傳圖片


def save_credentials_s():  # 帳密_s
    with open("credentials_s.txt", "w") as f:
        for username, password in valid_credentials_s.items():
            f.write(f"{username}:{password}\n")


def load_credentials_s():
    try:
        with open("credentials_s.txt", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                valid_credentials_s[parts[0]] = parts[1]
    except FileNotFoundError:
        print("No existing credentials file found.")


def authenticate_s(username, password):
    load_credentials_s()
    if username in valid_credentials_s and valid_credentials_s[username] == password:
        return True
    else:
        return False


def update_credentials_s(username, password):
    valid_credentials_s[username] = password
    save_credentials_s()


def save_credentials_t():  # 帳密_t
    with open("credentials_t.txt", "w") as f:
        for username, password in valid_credentials_t.items():
            f.write(f"{username}:{password}\n")


def load_credentials_t():
    try:
        with open("credentials_t.txt", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                valid_credentials_t[parts[0]] = parts[1]
    except FileNotFoundError:
        print("No existing credentials file found.")


def authenticate_t(username, password):
    load_credentials_t()
    if username in valid_credentials_t and valid_credentials_t[username] == password:
        return True
    else:
        return False


def update_credentials_t(username, password):
    valid_credentials_t[username] = password
    save_credentials_t()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/logins', methods=['GET', 'POST'])  # 帳密學生
def logins():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if authenticate_s(username, password):
            return render_template('welcomes.html', username=username)
        else:
            return "失敗"

    return render_template('logins.html')


@app.route('/logint', methods=['GET', 'POST'])  # 帳密老師
def logint():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if authenticate_t(username, password):
            return render_template('welcomet.html', username=username)
        else:
            return "失敗"

    return render_template('logint.html')


@app.route('/other_page1', methods=['GET', 'POST'])  # 註冊
def signupt():
    if request.method == 'POST':
        username = request.form['username2']
        password = request.form['password2']

        if username in valid_credentials_t:
            return "帳號已被註冊，註冊失敗"
        else:
            valid_credentials_t.update({username: password})
            return render_template('logint.html', username=username)

    return render_template('p1.html')


@app.route('/other_page2', methods=['GET', 'POST'])  # 註冊
def signups():
    if request.method == 'POST':
        username = request.form['username2']
        password = request.form['password2']

        if username in valid_credentials_s:
            return "帳號已被註冊，註冊失敗"
        else:
            valid_credentials_s.update({username: password})
            return render_template('logins.html', username=username)

    return render_template('p2.html')


@app.route('/uploadcustomfile', methods=['POST'])  # 上傳
def upload_image():
    if 'sendfile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['sendfile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded')
        return redirect(url_for('other_welcomet'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)


# 勿更動位置 ######################################################
client1 = ComputerVisionClient(endpoint, credentials)
photos_folder_path = 'uploads'  # 路徑 # 勿更動位置


@app.route('/uploadcustomfile2', methods=['GET', 'POST'])  # 上傳
def upload_image2():
    description_text = None
    keywords = None
    image_data = None
    photo_files = [f for f in os.listdir(photos_folder_path) if os.path.isfile(
        os.path.join(photos_folder_path, f))]

    selected_file = request.form.get('selected_image')

    if request.method == 'POST' and selected_file:
        image_path = os.path.join(photos_folder_path, selected_file)

        with open(image_path, "rb") as image_stream:
            image_data = base64.b64encode(image_stream.read()).decode('utf-8')
            image_stream.seek(0)
            results = client1.analyze_image_in_stream(image_stream, visual_features=[
                VisualFeatureTypes.description, VisualFeatureTypes.tags])

        if results.description.captions:
            description_text = results.description.captions[0].text
            keywords = ", ".join(tag.name for tag in results.tags)

            # Excel
            excel_path = 'ImageAnalyzer.xlsx'  # 路徑
            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
                df_update = pd.DataFrame({'File Name': [selected_file], 'Description': [
                                         description_text], 'Keywords': [keywords]})
                df = df[df['File Name'] != selected_file]  # 避免重覆儲存
                df = pd.concat([df, df_update], ignore_index=True)
            else:
                df = pd.DataFrame({'File Name': [selected_file], 'Description': [
                                  description_text], 'Keywords': [keywords]})
            df.to_excel(excel_path, index=False)

        else:
            description_text = "No description found."

    return render_template('uploadImg2.html', photo_files=photo_files, description_text=description_text, keywords=keywords, image_data=image_data, selected_file=selected_file)


@app.route('/quiz.html')  # 對話
def quiz():
    return render_template('p4.html')


def step1():  # Step1
    # 設定路徑 #讀excel #清空Prompt列
    excel_file_path_image_caption = "ImageCaption.xlsx"
    image_Prompt_df = pd.read_excel(excel_file_path_image_caption)
    image_Prompt_df['Prompt'] = ""  # 清空Prompt列

    if not image_Prompt_df.empty:
        # ImageCaption.xlsx 中隨機抓取一行
        random_index = image_Prompt_df.sample(n=1).index[0]
        random_caption = image_Prompt_df.loc[random_index, 'Caption']

        user_message = {
            "role": "user", "content": f"Give me a situational description of between 20 words based on the following keywords: {random_caption}"}
        messages = [user_message]

        response = client.chat.completions.create(
            messages=messages,
            model="chat3",
            temperature=1,
            max_tokens=200
        )

        assistant_response = response.choices[0].message.content
        logger.info(f"User: {user_message['content']}")
        logger.info(f"Chat: {assistant_response}")

        # 存回 ImageCaption.xlsx 的對應的位置
        image_Prompt_df.loc[random_index, 'Prompt'] = str(assistant_response)
        image_Prompt_df.to_excel(excel_file_path_image_caption, index=False)

        logger.info(f"Prompt 已保存到 ImageCaption.xlsx 的 'Prompt' 列.")
        # 傳送封包，使網頁顯示
        emit('step1return', {'message': str(assistant_response)})


def generate_assistant_response(user_input):  # 只有step2跟對話有用到
    user_message = {"role": "user", "content": user_input}
    try:
        response = client.chat.completions.create(
            messages=[user_message],
            model="chat3",
            temperature=1,
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in generate_assistant_response: {e}")
        return f"Error: {e}"


def test2(data):  # Step2
    # 設定路徑 #讀excel #清空excel
    excel_file_path_dialogue_record = "dialogue_record.xlsx"
    dialogue_df = pd.read_excel(excel_file_path_dialogue_record)
    dialogue_df.drop(dialogue_df.index, inplace=True)  # 清空excel

    print("Situation:  "+data)
    assistant_response = generate_assistant_response(
        f"Talk to me based on the following situation:\n\"{data}\"\nYou speak a sentence first and wait for my answer, and every answer with a question.")
    # 傳送封包，使網頁顯示
    emit('assistant_message', {'message': assistant_response})
    # 存回 diallougue_record.xlsx 的對應的位置
    dialogue_df = pd.concat([dialogue_df, pd.DataFrame({'User': [""], 'Assistant': [
                            assistant_response]})], ignore_index=True)  # ignore_index不知道可以幹嘛先放著
    dialogue_df.to_excel(excel_file_path_dialogue_record,
                         index=False)  # index=False 代表不會有索引值


def step3():  # Step3
    excel_file_path_dialogue_record = "dialogue_record.xlsx"
    dialogue_df = pd.read_excel(excel_file_path_dialogue_record)
    user_sentences = dialogue_df['User'].tolist()
    excel_file_path_check_ck = "check_ck.xlsx"
    check_df = pd.read_excel(excel_file_path_check_ck)
    check_df.drop(check_df.index, inplace=True)  # 清空excel

    fixed_prompt = "Can you check the spelling and grammar in the following text? Tell me clearly if there is an error and where it is"
    # conversation_history = [{"role": "user", "content": f"{fixed_prompt} ({user_sentences[0]})"}]

    for user_sentence in user_sentences[1:]:
        prompt = [
            {"role": "user", "content": f"{fixed_prompt} ({user_sentence})"}]
        messages = prompt

        response = client.chat.completions.create(
            messages=messages,
            model="chat3",
            temperature=1,
            max_tokens=200
        )
        # conversation_history += prompt
        assistant_response = response.choices[0].message.content
        logger.info(f"Chat: {assistant_response}")
        # 傳送封包，使網頁顯示
        emit('step3return', {'message': str(assistant_response)})
        # 存回 check_ck.xlsx 的對應的位置
        check_df = pd.concat([check_df, pd.DataFrame({'User': [user_sentence], 'Assistant': [
                             assistant_response]})], ignore_index=True)  # ignore_index不知道可以幹嘛先放著
        check_df.to_excel(excel_file_path_check_ck,
                          index=False)  # index=False 代表不會有索引值


@socketio.on('connect')
def handle_connect():
    print('Client connected成功連接')

# 網頁按下send按鈕


@socketio.on('user_input')
def reply(data):
    excel_file_path_dialogue_record = "dialogue_record.xlsx"
    dialogue_df = pd.read_excel(excel_file_path_dialogue_record)
    assistant_response = generate_assistant_response(
        f"Talk to me based on the following sentence:\n\"{data}\"\nYou speak a sentence with a question.")

    emit('assistant_message', {'message': assistant_response})
    dialogue_df = pd.concat([dialogue_df, pd.DataFrame({'User': [data['content']], 'Assistant': [
                            assistant_response]})], ignore_index=True)  # ignore_index不知道可以幹嘛先放著
    dialogue_df.to_excel(excel_file_path_dialogue_record,
                         index=False)  # index=False 代表不會有索引值


@socketio.on('step1')
def runstep1():
    step1()


@socketio.on('step2')
def runstep2(data):
    test2(data["content"])


@socketio.on('step3')
def runstep3():
    step3()

# 對話


@socketio.on('initial_quiz')  # 測驗
def init():
    excel_file_path_quiz_mode = "quiz_mode.xlsx"
    quiz_mode_df = pd.read_excel(excel_file_path_quiz_mode)
    quiz_mode_df.drop(quiz_mode_df.index, inplace=True)  # 清空excel
    quiz_mode_df.to_excel(excel_file_path_quiz_mode,
                          index=False)  # index=False 代表不會有索引值


@socketio.on('give_question')
def give_question(data):
    if data['content'] == 6:
        emit('complete')
        return
    elif data['content'] <= 2:
        prompt = [{"role": "user", "content": "Ask a vocabulary question. The question is a multiple-choice question. Just say Question directly."}]
    else:
        prompt = [{"role": "user", "content": "Ask a vocabulary question. The question is a fill-in-the-blank question with options. Just say Question directly, don't say \"Question:\"."}]
    messages = prompt

    response = client.chat.completions.create(
        messages=messages,
        model="chat3",
        temperature=1,
        max_tokens=200
    )
    assistant_response = response.choices[0].message.content

    question = f"{data['content']}: {assistant_response}"
    emit('assistant_message', {
         'message': question, 'content': assistant_response})


@socketio.on('user_answer')
def mark_quiz(data):
    excel_file_path_quiz_mode = "quiz_mode.xlsx"
    quiz_mode_df = pd.read_excel(excel_file_path_quiz_mode)

    prompt = [{"role": "user", "content": f"Question is : {data['question']}. Student's answer is: {data['content']}. Judge whether the answer is correct. Just say \"correct\" or \"incorrect\"."}]
    messages = prompt

    response = client.chat.completions.create(
        messages=messages,
        model="chat3",
        temperature=1,
        max_tokens=200
    )
    assistant_response = response.choices[0].message.content
    correct = assistant_response

    quiz_mode_df = pd.concat([quiz_mode_df, pd.DataFrame({'User': [data['content']], 'Assistant': [
                             data['question']], 'Correct': [assistant_response]})], ignore_index=True)  # ignore_index不知道可以幹嘛先放著
    quiz_mode_df.to_excel(excel_file_path_quiz_mode,
                          index=False)  # index=False 代表不會有索引值

    emit('assistant_message', {'message': f"The answer is {correct}"})


@socketio.on('see_result')
def see_result():
    excel_file_path_quiz_mode = "quiz_mode.xlsx"
    quiz_mode_df = pd.read_excel(excel_file_path_quiz_mode)
    excel_file_path_correct_rate = "correct_rate.xlsx"
    correct_rate_df = pd.read_excel(excel_file_path_correct_rate)
    correct_counts = quiz_mode_df['Correct'].tolist()

    print(correct_counts[0:])

    for correct_count in correct_counts[0:]:
        print(correct_count)

    prompt = [{"role": "user", "content": f"Here is the correct list, {correct_counts[0:]}, tell me correct rate. Just tell me the percentage, not the calculation process."}]
    messages = prompt

    response = client.chat.completions.create(
        messages=messages,
        model="chat3",
        temperature=1,
        max_tokens=200
    )
    assistant_response = response.choices[0].message.content
    rate = assistant_response
    print(rate)

    taiwan_timezone = timezone(timedelta(hours=8))
    current_datetime_taiwan = datetime.now(
        taiwan_timezone).strftime("%Y-%m-%d %H:%M:%S")

    correct_rate_df = pd.concat([correct_rate_df, pd.DataFrame({'Date (Taiwan Time)': [
                                current_datetime_taiwan], 'Correct_Rate (%)': [rate]})], ignore_index=True)  # ignore_index不知道可以幹嘛先放著
    correct_rate_df.to_excel(excel_file_path_correct_rate,
                             index=False)  # index=False 代表不會有索引值

    emit('result', {'message': rate})


@app.route('/other_logint')  # 首頁
def other_logint():
    return render_template('logint.html')


@app.route('/other_logins')  # 首頁
def other_logins():
    return render_template('logins.html')


@app.route('/other_page1')  # 首頁
def other_page1():
    return render_template('p1.html')


@app.route('/other_page2')  # 首頁
def other_page2():
    return render_template('p2.html')


@app.route('/other_page3')  # welcomes
def other_page3():
    return render_template('p3.html')


@app.route('/other_page4')  # welcomes
def other_page4():
    return render_template('p4.html')


@app.route('/other_welcome')  # 返回
def other_welcome():
    return render_template('welcomes.html')


@app.route('/other_welcomet')  # 老師
def other_welcomet():
    return render_template('uploadImg.html')


@app.route('/ImageAnalyzer')  # 圖片描述
def ImageAnalyzer():
    return render_template('uploadImg2.html')


if __name__ == '__main__':
    dialogue_df = pd.DataFrame(columns=['User', 'Assistant'])

    socketio.run(app, debug=True)
