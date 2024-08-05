import json
import requests
import os
from openai import OpenAI
import openai
from prompts import assistant_instructions


#注意你复制完可能要加上Bearer这个前缀
AIRTABLE_API_KEY = 'Bearer patn0TbFXegKQHiXx.c97a5c2100ea9845f2dea7ef47eaf972eeb73ff21524c274f71877f866279961'
OPENAI_API_KEY = 'sk-Cz3qhY1gwI5n4u48F4387eD62fEe427dB02fFdB1EdC6048'
#OPENAI_API_BASE ='https://api.novelnetwork.online/v1'
# Init OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)#,base_url=OPENAI_API_BASE)


# 创建一个表单数据
def create_lead(name, phone, wechat,address,summary,intention):
  url = "https://api.airtable.com/v0/app3aq30IUpiJ3XhL/tblCgaCOreclAYAbH"  # 替换自己的
  headers = {
      "Authorization": AIRTABLE_API_KEY,
      "Content-Type": "application/json"
  }
  data = {
      "records": [{
          "fields": {
              "Name": name,
              "Phone": phone,
              "Wechat": wechat,
              "Address": address,
              "Summary": summary,
              "Intention": intention
          }
      }]
  }
  response = requests.post(url, headers=headers, json=data)
  if response.status_code == 200:
    print("Lead created successfully.")
    return response.json()
  else:
    print(f"Failed to create lead: {response.text}")

# 创建助手
def create_assistant(client):
  assistant_file_path = 'assistant.json'

  # 如果json不存在就创建一个，如果换本地知识库要删除原来的
  if os.path.exists(assistant_file_path):
    with open(assistant_file_path, 'r') as file:
      assistant_data = json.load(file)
      assistant_id = assistant_data['assistant_id']
      print("Loaded existing assistant ID.")
  else:
    file = client.files.create(file=open("dige.txt", "rb"),
                               purpose='assistants')

    assistant = client.beta.assistants.create(
        # 这里面要调用你写好的指令和接下来会用到的API
        instructions=assistant_instructions,
        model="gpt-4-1106-preview",
        tools=[
            {
                "type": "retrieval"  # This adds the knowledge base as a tool
            },
            {
                "type": "function",  # 写表单
                "function": {
                    "name": "create_lead",
                    "description":
                    "Capture lead details and save to Airtable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the lead."
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number of the lead."
                            },
                            "wechat": {
                                "type": "string",
                                "description": "Phone number of the lead."
                            },
                            "address": {
                                "type": "string",
                                "description": "Address of the lead."
                            },
                            "summary": {
                                "type": "string",
                                "description": "Summarize students’ questions."
                            },
                            "intention": {
                                "type": "string",
                                "description": "Students’ purchase intention."
                            }
                        },
                        "required": ["name", "phone","wechat","address","summary","intention"]
                    }
                }
            }
        ],
        file_ids=[file.id])

    # 要创建一个新的json
    with open(assistant_file_path, 'w') as file:
      json.dump({'assistant_id': assistant.id}, file)
      print("Created a new assistant and saved the ID.")

    assistant_id = assistant.id

  return assistant_id
