from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END

from langchain_community.chat_models.tongyi import ChatTongyi
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

model_with_tools = ChatTongyi(
    model="qwen-max",
    temperature=0,
    api_key=""
).bind_tools(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# # example with a single tool call
# for chunk in app.stream(
#     {"messages": [("human", "what's the weather in sf?")]}, stream_mode="values"
# ):
#     chunk["messages"][-1].pretty_print()

# example with a multiple tool calls in succession

for chunk in app.stream(
    {"messages": [("human", "what's the weather in the coolest cities?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

"""
<IPython.core.display.Image object>
================================ Human Message =================================

what's the weather in the coolest cities?
================================== Ai Message ==================================
Tool Calls:
  get_coolest_cities (call_f99d9bb92a3f4f1a998b70)
 Call ID: call_f99d9bb92a3f4f1a998b70
  Args:
================================= Tool Message =================================
Name: get_coolest_cities

nyc, sf
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_85b87781f57744aab8cd96)
 Call ID: call_85b87781f57744aab8cd96
  Args:
    location: nyc
================================= Tool Message =================================
Name: get_weather

It's 90 degrees and sunny.
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_6a88f9f05427443991bd4c)
 Call ID: call_6a88f9f05427443991bd4c
  Args:
    location: sf
================================= Tool Message =================================
Name: get_weather

It's 60 degrees and foggy.
================================== Ai Message ==================================

The weather in the coolest cities is as follows:

- In New York City (NYC), it's 90 degrees and sunny.
- In San Francisco (SF), it's 60 degrees and foggy.

"""