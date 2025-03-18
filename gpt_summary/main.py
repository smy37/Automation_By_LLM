import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import json
from datetime import datetime, timezone, timedelta
import openai
import tiktoken
from common.utils import doc_split

EMBEDDING_MODEL = "text-embedding-3-large"

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

data_path = BASE_DIR + r"/conversations.json"
json_data = json.load(open(data_path))
print(len(json_data))

data_by_time = {}
for i in range(len(json_data)): ## 대화방 단위
    mapping = json_data[i].get("mapping")
    create_time = json_data[i].get("create_time")

    date_str= get_korea_time(create_time)
    if date_str not in data_by_time:
        data_by_time[date_str] = []
    temp = ""
    for k in mapping:   ## 대화말 단위
        if mapping[k].get("message"):
            role = mapping[k].get("message").get("author")["role"]
            if mapping[k].get("message").get("content").get("parts"):
                msg = mapping[k].get("message").get("content").get("parts")[0]
                if len(msg) >0 and msg:
                    temp += f"{role}: {mapping[k].get("message").get("content").get("parts")[0]}\n\n"
            else:
                temp += f"{role}: {mapping[k].get("message").get("content").get("text")}\n\n"
    data_by_time[date_str].append(temp)       

cnt = []
for day in data_by_time:
    for chat_room in data_by_time[day]:
        chat_room = chat_room.replace('<|endoftext|>', '')
        print(count_gpt4o_tokens(chat_room))
        cnt.append(count_gpt4o_tokens(chat_room))

print(sorted(cnt, reverse=True))
