import sys
import os
import json
from datetime import datetime, timezone, timedelta
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
def get_korea_time(utc_time):
    ### 날짜 가지고 오기.
    KST = timezone(timedelta(hours=9))
    dt = datetime.fromtimestamp(utc_time, tz=KST)
    date_str = dt.strftime('%Y-%m-%d')
    weekday_str = dt.strftime('%A')

    return date_str+ f"-{weekday_str}"

path = r"C:\Users\seong\OneDrive\바탕 화면\project\GPT_Project\gpt_summary\conversations.json"
json_data = json.load(open(path))
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
    print(data_by_time)
    break