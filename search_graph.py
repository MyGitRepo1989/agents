import langchain_tavily 
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import io
import os
from dotenv import load_dotenv


model = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
memory = MemorySaver()

#THIS IS YOUR SEARCH API KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

#REST IS SAME FOR ANY LLM GRAPH USE THIS TOOL