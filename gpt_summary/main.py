import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import json
from datetime import datetime, timezone, timedelta
import openai
import tiktoken
from common.openai_api import ask_gpt
from common import prompt_lib
from common.multi_thread import process_multi_thread
from pydantic import BaseModel
from common import utils
from common import variable

openai.api_key = os.getenv("OPENAI_API_KEY")
def get_korea_time(utc_time):
    ### 날짜 가지고 오기.
    KST = timezone(timedelta(hours=9))
    dt = datetime.fromtimestamp(utc_time, tz=KST)
    date_str = dt.strftime('%Y-%m-%d')
    weekday_str = dt.strftime('%A')

    return date_str+ f"-{weekday_str}"

def count_gpt4o_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class output_format(BaseModel):
    markdown_summary: str

def summary_process(temp_data, max_tokens=16384, model_name="gpt-4o", temperature=0.7, reasoning_effort="medium"):
    resp = ask_gpt(temp_data, prompt_lib.summary_prompt, output_format, max_tokens, model_name, temperature, reasoning_effort).markdown_summary
    return resp


if __name__ == "__main__":
    b_multi_thread = True
    data_path = BASE_DIR + r"/conversations.json"
    json_data = json.load(open(data_path))
    write_path_summary = "./result/summary"
    write_path_concat = "./result/concat"
    INPUT_MAX_TOKEN = 16384
    print(len(json_data))

    data_by_time = {}
    for i in range(len(json_data)): ## 대화방 단위
        mapping = json_data[i].get("mapping")
        create_time = json_data[i].get("create_time")

        date_str= get_korea_time(create_time)
        if date_str not in data_by_time:
            data_by_time[date_str] = []
        temp_txt_l = []
        for k in mapping:   ## 대화말 단위
            if mapping[k].get("message"):
                role = mapping[k].get("message").get("author")["role"]
                if mapping[k].get("message").get("content").get("parts"):
                    msg = mapping[k].get("message").get("content").get("parts")[0]
                    if len(msg) >0 and msg:
                        temp_txt_l.append(f"{role}: {msg}\n\n")
                else:
                    txt_msg = mapping[k].get("message").get("content").get("text")
                    temp_txt_l.append(f"{role}: {txt_msg}\n\n")
        temp_txt_join = "".join(temp_txt_l)

        data_by_time[date_str].append(temp_txt_join)



    sum_token = 0


    for day in data_by_time:
        if day.replace("-", "_")+".md" in os.listdir(write_path_summary):   ## 이미 있는 날짜의 파일이면 다시 생성하지 않음.
            continue
        print(day)
        f_name = day.replace("-", "_")+".md"
        temp = []
        temp_datas = []
        temp_token_num = 0      ## 최대 인풋 토큰 넘는 것을 방지하기 위해 카운팅
        chunk_num = 1
        for chat_room in data_by_time[day]:
            chat_room = chat_room.replace('<|endoftext|>', '')
            token_num = count_gpt4o_tokens(chat_room)
            sum_token += token_num
            temp_token_num += token_num

            if b_multi_thread:
                if temp_token_num > INPUT_MAX_TOKEN:
                    chunks = utils.doc_split(chat_room, variable.EMBEDDING_MODEL, INPUT_MAX_TOKEN)
                    for chunk in chunks:
                        temp_datas.append(chunk)
                else:
                    temp_datas.append(chat_room)
            else:
                if temp_token_num > INPUT_MAX_TOKEN:
                    chunks = utils.doc_split(chat_room, variable.EMBEDDING_MODEL, INPUT_MAX_TOKEN)
                    for chunk in chunks:
                        temp.append(ask_gpt(chunk, prompt_lib.summary_prompt, output_format).markdown_summary)
                else:
                    temp.append(ask_gpt(chat_room, prompt_lib.summary_prompt, output_format).markdown_summary)
        print(len(temp_datas))
        after_data = process_multi_thread(temp_datas, summary_process, 16384, "gpt-4o", 0.7, "medium") if b_multi_thread else temp

        with open(os.path.join(write_path_summary, f_name), 'w', encoding='utf-8') as wr:
            wr.write("\n########## New Chat ##########\n".join(after_data))
        with open(os.path.join(write_path_concat, f_name.replace(".md", ".txt")), 'w', encoding='utf-8-sig') as wr:
            wr.write("\n########## New Chat ##########\n".join(temp_datas))

    print("예상가격(달러):", sum_token/1000000*5)