# coding:utf-8
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
            },
            "required": ["location"],
        },
    },
]

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the weather like in San Francisco?"),
]
chatLLM = ChatTongyi(
    model="qwen-max",
    temperature=0,
    api_key="")
# llm_kwargs = {"tools": tools, "result_format": "message"}
# ai_message = chatLLM.bind(**llm_kwargs).invoke(messages)
# print(ai_message)
"""
content='' additional_kwargs={'tool_calls': [{'function': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco"}'}, 
'index': 0, 'id': 'call_3f2c203771954269884012', 'type': 'function'}]} 
response_metadata={'model_name': 'qwen-max', 'finish_reason': 'tool_calls', 'request_id': '4af00518-c795-98cb-a19a-a67c0a73ee63', 
'token_usage': {'input_tokens': 229, 'output_tokens': 19, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 248}} 
id='run-21093d5e-6fc5-4a1d-b1fc-7fe80c660bda-0' 
tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'San Francisco'}, 'id': 'call_3f2c203771954269884012', 'type': 'tool_call'}]

"""
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


llm = ChatTongyi(model="qwen-turbo", api_key="sk-064b2c7a65b9478aab4d263c7bf7bdb9")

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")

print(msg)