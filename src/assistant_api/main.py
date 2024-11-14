import json
import os
import time
from flask import Flask, request, jsonify
import openai
from openai import OpenAI
import functions

from packaging import version

required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)
OPENAI_API_KEY = 'sk-xxx'
#OPENAI_API_BASE ='https://api.novelnetwork.online/v1'
if current_version < required_version:
  raise ValueError(
      f"Error: OpenAI version {openai.__version__} is less than the required version 1.1.1"
  )
else:
  print("OpenAI version is compatible.")

# Flask常规操作
app = Flask(__name__)

# KEY写自己的
client = OpenAI(api_key=OPENAI_API_KEY)#,base_url=OPENAI_API_BASE)

# 加载助手
assistant_id = functions.create_assistant(
    client)  # this function comes from "functions.py"


# 得到对话请求
@app.route('/start', methods=['GET'])
def start_conversation():
  print("Starting a new conversation...")
  thread = client.beta.threads.create()
  print(f"New thread created with ID: {thread.id}")
  return jsonify({"thread_id": thread.id})


# 调用GPT和API产生结果
@app.route('/chat', methods=['POST'])
def chat():
  data = request.json
  thread_id = data.get('thread_id')
  user_input = data.get('message', '')

  if not thread_id:
    print("Error: Missing thread_id")
    return jsonify({"error": "Missing thread_id"}), 400

  print(f"Received message: {user_input} for thread ID: {thread_id}")


  client.beta.threads.messages.create(thread_id=thread_id,
                                      role="user",
                                      content=user_input)

  run = client.beta.threads.runs.create(thread_id=thread_id,
                                        assistant_id=assistant_id)

  # 是否需要调用API
  while True:
    run_status = client.beta.threads.runs.retrieve(thread_id=thread_id,
                                                   run_id=run.id)
    # print(f"Run status: {run_status.status}")
    if run_status.status == 'completed':
      break
    elif run_status.status == 'requires_action':
      # 处理各种API
      for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
        if tool_call.function.name == "create_lead":
          # Process lead creation
          arguments = json.loads(tool_call.function.arguments)
          output = functions.create_lead(arguments["name"], arguments["phone"],
                                          arguments["wechat"], arguments["address"], arguments["summary"], arguments["intention"])
          client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id,
                                                       run_id=run.id,
                                                       tool_outputs=[{
                                                           "tool_call_id":
                                                           tool_call.id,
                                                           "output":
                                                           json.dumps(output)
                                                       }])
      time.sleep(1)  # Wait for a second before checking again

  # 返回结果
  messages = client.beta.threads.messages.list(thread_id=thread_id)
  response = messages.data[0].content[0].text.value

  print(f"Assistant response: {response}")
  return jsonify({"response": response})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
