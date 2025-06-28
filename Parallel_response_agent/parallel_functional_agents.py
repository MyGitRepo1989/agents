# !! THIS PROGRAM ILLUSTRATES PARALLEL PROCESSING USING FUNCTIONAL API- Translates a sentence into multple languages at one go
# - User inputs sentence
# - we have 8 parallel translation processes
# - all responses are send back
# -- we dont create a graph here


from typing_extensions import TypedDict
import random
from typing import Literal
from typing_extensions import Literal

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.func import entrypoint, task
import json




llm = ChatOpenAI(model="gpt-4o", temperature=0)

def outside_process(some_text):
    add_this ="BASIC JOKE "
    final_message = add_this + str(some_text)
    return final_message
    

#PARALLEL TASK 1
@task
def call_llm_1(topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {topic}")
    final_1 = outside_process(msg.content)
    return final_1


#PARALLEL TASK 2
@task
def call_llm_2(topic:str):
    """This is the 2nd llm making joke in a mean rude way generate mean_joke\
        use cruel words and be mean """
    msg = llm.invoke(f"write the joke in a mean rude way use funny cruel words and be mean {topic}")
    return msg.content


#PARALLEL TASK 3
@task
def call_llm_3(topic:str):
    """This is the 3rd llm making joke in a loud over done way generate loud_joke\
        add strong word and capatalization and be agressive """
    msg = llm.invoke(f"write the joke in a loud way add capatalization and act out {topic}")
    return msg.content

#PARALLEL TASK 4
@task
def call_llm_4(topic:str):
    """This is the 4th llm making joke like a seasoned politicial generate politician_joke """
    msg = llm.invoke(f"write the joke like a politician who is a big liar {topic}")
    return msg.content

#PARALLEL TASK 5
@task
def call_llm_5(topic:str):
    """This is the 5th llm making joke like a gym motvational guru \
        add quote type to encourage action and strength """
    msg = llm.invoke(f"write the joke as if you are a gym instructor and a gym buff always pushing \
        others with quotes to do action and workout{topic}")
    return msg.content



#THE FINAL AGGREGATOR
@task
def aggregator(topic, joke, mean_joke,loud_joke,politician_joke,gym_joke):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {topic}!\n\n"
    combined += f"TASK 1- BASIC JOKE:\n{joke}\n\n"
    combined += f"TASK 2- MEAN JOKE:\n{mean_joke}\n\n"
    combined += f"TASK 3- LOUD JOKE:\n{loud_joke}\n\n"
    combined += f"TASK 4- POLITICIAN JOKE:\n{politician_joke}\n\n"
    combined += f"TASK 5- GYM JOKE:\n{gym_joke}\n\n"
    return combined



# Build workflow
@entrypoint()
def parallel_workflow(topic: str):
    basic_joke = call_llm_1(topic)
    mean_joke = call_llm_2(topic)
    loud_joke = call_llm_3(topic)
    politician_joke = call_llm_4(topic)
    gym_joke = call_llm_5(topic)
    
    return aggregator(
        topic, basic_joke.result(), mean_joke.result(),loud_joke.result(),politician_joke.result(),gym_joke.result()
    )

# Invoke
user_topic= input ("add topic of the joke you want:  ")
for step in parallel_workflow.stream({user_topic}, stream_mode="updates"):
    print(step)
    print("\n")