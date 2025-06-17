# pip install --quiet -U langgraph langchain_anthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"


tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

print(tool_node.invoke({"messages": [message_with_single_tool_call]}))

# {'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id')]}

message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_coolest_cities",
            "args": {},
            "id": "tool_call_id_1",
            "type": "tool_call",
        },
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id_2",
            "type": "tool_call",
        },
    ],
)

print(tool_node.invoke({"messages": [message_with_multiple_tool_calls]}))
# {'messages': [ToolMessage(content='nyc, sf', name='get_coolest_cities', tool_call_id='tool_call_id_1'), ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id_2')]}

from typing import Literal
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


from langchain_community.chat_models.tongyi import ChatTongyi


model_with_tools = ChatTongyi(
    model="qwen-max",
    temperature=0,
    api_key=""
).bind_tools(tools)

print(model_with_tools.invoke("what's the weather in sf?").tool_calls)
print(model_with_tools.invoke("what's the weather in sf?"))

# [{'name': 'get_weather', 'args': {'location': 'San Francisco, US'}, 'id': 'call_b15d3f9d703f48cda0b3bc', 'type': 'tool_call'}]

print(tool_node.invoke({"messages": [model_with_tools.invoke("what's the weather in sf?")]}))
# {'messages': [ToolMessage(content="It's 90 degrees and sunny.", name='get_weather', tool_call_id='call_2e76d6803f9f4143b60d50')]}
