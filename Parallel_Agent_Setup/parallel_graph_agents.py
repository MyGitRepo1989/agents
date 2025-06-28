#THIS PROGRAM ILLUSTRATES PARALLEL PROCESSING  - 
# Translates a sentence into multple languages at one go
# - User inputs sentence
# - we have 8 parallel translation processes
# - all responses are send back


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

llm = ChatOpenAI(model="gpt-4o", temperature=0)


#THIS IS YOUR STATE 
# Graph state
class State(TypedDict):
    sentence: str
    english: str
    french: str
    german: str
    greek:str
    hindi: str
    chinese : str
    urdu : str
    japanese : str
    combined_output: str
    
#THE LLM THAT CHECKS THE WORK
def double_check(original, translation):
    msg = llm.invoke(f"please imporve this translation : original is {original}  and translation is {translation}")
    return msg.content

# Nodes
def call_llm_1(state: State):
    """First LLM call to translate sentence to english"""

    msg = llm.invoke(f"Translate this sentence in english language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"english": msg.content}


def call_llm_2(state: State):
    """First LLM call to translate sentence to french"""

    msg = llm.invoke(f"Translate this sentence in french language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"french": msg.content}


def call_llm_3(state: State):
    """First LLM call to translate sentence to german"""

    msg = llm.invoke(f"Translate this sentence in german language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"german": msg.content}


def call_llm_4(state: State):
    """First LLM call to translate sentence to greek"""

    msg = llm.invoke(f"Translate this sentence in greek language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"greek": msg.content}


def call_llm_5(state: State):
    """First LLM call to translate sentence to hindi"""

    msg = llm.invoke(f"Translate this sentence in hindi language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"hindi": msg.content}


def call_llm_6(state: State):
    """First LLM call to translate sentence to chinese"""

    msg = llm.invoke(f"Translate this sentence in chinese language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"chinese": msg.content}


def call_llm_7(state: State):
    """First LLM call to translate sentence to urdu"""

    msg = llm.invoke(f"Translate this sentence in urdu language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"urdu": msg.content}


def call_llm_8(state: State):
    """First LLM call to translate sentence to japanese"""

    msg = llm.invoke(f"Translate this sentence in japanese language {state['sentence']}")
    
    original = state['sentence']
    transalation = msg.content
    msg.content= "IMPOROVED : " + double_check(original, transalation)
    
    return {"japanese": msg.content}



def aggregator(state: State):
    """Combine the 8 translations into a single output"""

    combined = f"Here's all the translations of this sentence about {state['sentence']}!\n\n"
    combined = f"ORIGINAL SENTENCE {state['sentence']}\n\n"
    
    combined += f"ENGLISH:\n{state['english']}\n\n"
    combined += f"FRENCH:\n{state['french']}\n\n"
    combined += f"GERMAN:\n{state['german']}\n\n"
    combined += f"GREEK:\n{state['greek']}\n\n"
    combined += f"HINDI:\n{state['hindi']}\n\n"
    combined += f"CHINESE:\n{state['chinese']}\n\n"
    combined += f"URDU:\n{state['urdu']}\n\n"
    combined += f"JAPANESE:\n{state['japanese']}\n\n"
    
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("agent_wf1", call_llm_1)
parallel_builder.add_node("agent_wf2", call_llm_2)
parallel_builder.add_node("agent_wf3", call_llm_3)
parallel_builder.add_node("agent_wf4", call_llm_4)
parallel_builder.add_node("agent_wf5", call_llm_5)
parallel_builder.add_node("agent_wf6", call_llm_6)
parallel_builder.add_node("agent_wf7", call_llm_7)
parallel_builder.add_node("agent_wf8", call_llm_8)

parallel_builder.add_node("aggregator", aggregator)


# Add edges to connect nodes
parallel_builder.add_edge(START, "agent_wf1")
parallel_builder.add_edge(START, "agent_wf2")
parallel_builder.add_edge(START, "agent_wf3")
parallel_builder.add_edge(START, "agent_wf4")
parallel_builder.add_edge(START, "agent_wf5")
parallel_builder.add_edge(START, "agent_wf6")
parallel_builder.add_edge(START, "agent_wf7")
parallel_builder.add_edge(START, "agent_wf8")



parallel_builder.add_edge("agent_wf1", "aggregator")
parallel_builder.add_edge("agent_wf2", "aggregator")
parallel_builder.add_edge("agent_wf3", "aggregator")
parallel_builder.add_edge("agent_wf4", "aggregator")
parallel_builder.add_edge("agent_wf5", "aggregator")
parallel_builder.add_edge("agent_wf6", "aggregator")
parallel_builder.add_edge("agent_wf7", "aggregator")
parallel_builder.add_edge("agent_wf8", "aggregator")


parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

sentence = input("ADD SENTENCE THAT YOU WANT TRANSLATED: ")
state = parallel_workflow.invoke({"sentence": sentence})
print(state["combined_output"])


 #save the graph diagram   
from PIL import Image
with open("parallel_workflow.png", "wb") as f: f.write(parallel_workflow.get_graph().draw_mermaid_png())