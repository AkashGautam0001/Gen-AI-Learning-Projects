from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state:State):
    print("\nChatbot node executed", state)
    response = llm.invoke(state.get("messages"))
    return { "messages" : [response]}

def samplenode(state:State):
    print("\nSamplenode node executed", state)
    return { "messages" : ["Hi, This is a message from samplenode"]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)

def compile_graph_with_checkpinter():  
    client = MongoClient("mongodb://localhost:27017")
    checkpointer = MongoDBSaver(client)
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

graph = compile_graph_with_checkpinter()

config = {
    "configurable": {
        "thread_id": "1",
    }
}

updated_state = graph.invoke(State({"messages": ["What is my name ?"]}), config=config)


print("\nUpdated State: ", updated_state)