import os
import io
import sys
import random
import threading
import dotenv
import pandas as pd
from dotenv import load_dotenv
from gtts import gTTS

import streamlit as st
import streamlit.web.cli as stcli

from typing import Literal, Annotated
from typing_extensions import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools.tavily_search import TavilySearchResults

from IPython.display import Image, display

# ElevenLabs
import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Load environment variables
dotenv.load_dotenv()




# Initialize client with your API key
client = ElevenLabs(
    api_key="ELEVEN_LAB_KEY"  # Replace with your real key or move to env for security
)
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")
model = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
memory = MemorySaver()
door_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Schema for structured output to use as routing logic
class service_route(BaseModel):
    step: Literal["lock_door", "refills", "wifi", "reception", "other"] = Field(
        None, description="what service the guest wants from the AI Home System"
    )




# Augment the LLM with schema for structured output
home_router = llm.with_structured_output(service_route)

# Schema for structured output to use as routing logic
class which_door_route(BaseModel):
    step: Literal["front", "bedroom", "balcony", "garage", "not_found"] = Field(
        None, description="find the name of the door"
    )

# Augment the LLM with schema for structured outputs
door_router = door_llm.with_structured_output(which_door_route)

# Schema for structured output to use as routing logic
class find_refill_route(BaseModel):
    step: Literal["snacks", "tea_coffee", "drinks", "bread", "water", "not_found"] = Field(
        None, description="find the name of the refills"
    )

# Augment the LLM with schema for structured outputs
refill_router = door_llm.with_structured_output(find_refill_route)

# Define the state
class OverallState(TypedDict):
    service_sasked: str
    subjects: list
    llm_response : str = ""
    messages : str = ""
    greeting : str = ""
    door_name: str= ""
    last_message: str = ""
    refill_name: str= ""
   
# Define the graph
graph_builder = StateGraph(OverallState)



# Define the nodes
def process_input(state: OverallState):     
    if 'messages' in state and state['messages']:        
            
        user_input = state['messages']
        print("user_input", user_input)
        
        home_question ="You are a friendly guest service assistant in a building. \
            Greet the guest briefly and ask in one natural line what they need \
                help with: locking the door, ordering refills, Wi-Fi info, or calling \
                    reception. Keep it modern, clear, short, natural greeting in 1 to 1.5 lines max—no emojis "
        response = model.invoke([HumanMessage(content= str( user_input + home_question) )]).content   
        print(response, "RESPONSE")
        print("STATE", state)
        state['llm_response'] = response
        print("STATE 2 ", state)    
        print(state['llm_response'])
        state["greeting"] ="not_done"
        state["last_message"] = state["messages"]
    else:
        state['llm_response'] = "No messages to process."
    return state





def which_service(state:OverallState):   
    # Run the augmented LLM with structured output to serve as routing logic
    # ROUTE TO CHECK WHICH SERVICE
    service_decision = home_router.invoke(
        [
            SystemMessage(
                content="Route the service request from guest, lock door,  refills , \
                    wifi, reception or if its anything else return other."
            ),
            HumanMessage(content=state['messages']),
        ]
    )   
    user_request= service_decision.step    
    if  user_request == "lock_door":
        return "which_door"
    if  user_request == "wifi":
        return "wifi_info"
    if  user_request == "reception":
        return "reception"
    if  user_request == "refills":
        return "which_refill"
    if  user_request == "other":
        return "ask_service_again"


def which_door(state:OverallState):
    #ROUTE TO CHECK WHICH DOOR
    state['llm_response'] = model.invoke(" you are a guest service assistant ask the guest in a natural short way \
        what door they want locked front , bedroom , balcony , garage , you can include in the message \
            what they spoke last to seem natural" + state ["messages"]).content
        
    state["greeting"] = "done"
    
    door_asked= door_router.invoke(
        [
            SystemMessage(
                content= "find name of door front, bedroom , garage, balcony, and if not found return not_found"
            ),
            HumanMessage(content=state['messages']),
        ]
    ) 
    door_name = door_asked.step
    state["door_name"] = door_name
    print("FOUND NAME OF DOOR", door_name)

    return state


def find_door(state:OverallState):
    #ROUTE TO DIRECT TO DOOR ROUTER

    door_name = state["door_name"] 
    if  door_name == "front":
        return "door_locked_message" 
    if  door_name == "bedroom":
        return "door_locked_message" 
    if  door_name == "balcony":
        return "door_locked_message" 
    if  door_name == "garage":
        return "door_locked_message" 
    if  door_name == "not_found":
        return "ask_door_name" 
    


def door_locked_message(state:OverallState):
    #ROUTE SEND CONFIRMATION TO USER THAT DOOR IS LOCKED
    print ("PRINT I AM IN  LOCK DOOR MESSAGEs" , state)
    confirm_door_message  =  model.invoke(f'write this smart home system 1 line message to guest\
        the message is for locked door successfully with the door name provided\
            {state["door_name"]}').content
    state["llm_response"] = confirm_door_message 
    return state


def ask_door_name(state:OverallState):
    #ROUTE GENERIC DOOR ASK
    model_generated = "Please tell me which door -  front door, balcony door, garage door , bedroom door or anything else"
    state["llm_response"] = model_generated
    return state



def which_refill(state:OverallState):
   #ROUTE RO ASK WHICH REFILL GUEST WANTS
    state['llm_response'] = model.invoke("you are guest service - write a 1 line natural \
        1 line question for guest asking what\
        options they want refilled in condo : Snacks, Drinks, Tea & Coffee, Bread & Cheese, Water ? ").content
    print(" STATE REFILL", state)
    name_of_refill = refill_router.invoke(
        [
            SystemMessage(
                content= "find name of refills guest asked for Water, Snacks, Drinks, Tea & Coffee, Bread & Cheese\
                    and if nothing say not_found"
            ),
            HumanMessage(content=state['messages']),
        ]
    ) 
    found_refill = name_of_refill.step
    state["refill_name"] = found_refill
    print("THIS IS THE REFILL GUEST ASKED FOR ", found_refill, state["refill_name"])
    return state


def find_refill(state:OverallState):
    #ROUTE TO THE CORRECT REFILL NODE
    print("STATE", state)
    refill_asked_for = state["refill_name"] 
    if  refill_asked_for == "snacks":
        return "summary"
    if  refill_asked_for == "tea_coffee":
        return "summary"
    if  refill_asked_for == "drinks":
        return "summary" 
    if  refill_asked_for == "bread":
        return "summary"
    if  refill_asked_for == "water":
        return "summary" 
    if  refill_asked_for == "not_found":
        return "ask_refill_again" 



def ask_refill_again(state: OverallState):
    #THIS IS TO ASK REFILL AGAIN 
    print("STATE", state)
    state['llm_response'] = model.invoke(" The gues asked for refill and maybe did not tell us clearly so ask again\
        In as short and natural way - Hey tell me what refills u need \
        Snacks, Water , Drinks , Tea Coffee, Maybe some Bread " + \
            " you can include in the message \
            what they spoke last to seem natural" + state ["messages"]).content
    return state  # or any valid node name you *do* have defined




def summary(state:OverallState):
    #THIS IS FOR SUMMARY
    print("STATE", state)
    summary = model.invoke(f' send a 1 line courtsey message to condo guest that\
        their requests is confirmed. \
        The last message from guest was {state["messages"]} when we ask if they\
        had a request for refills for Snack Drinks etc. You can add their last reply to sound natural\
            your reply should be 1 line').content
    state['llm_response'] = summary
    return state


def wifi_info(state:OverallState):
    #THIS IS FOR WIFI INFORMATION
    print("STATE", state)
    wifi_details= " wifi name is BuildingWifi and password is 98Enjoy"
    state['llm_response'] = model.invoke(f"Inform guestin 1 line the wifi details {wifi_details}" \
                                        + "if needed their last message was , use it to seem natural "  + state["messages"]).content
    return state

def reception(state:OverallState):
    #THIS IS FOR RECEPTION
    print("STATE", state)
    state['llm_response'] = model.invoke("send a short 1 line message to guest 1 line -I have requested reception to contact you Thank  you" +\
        "for continuity you can if needed add what they said last" + state["messages"]).content
    return state



#THIS IS ALL OTHER GREETINGS
def ask_service_again(state: OverallState):
    greeting1 = " You are a smart guest service assistant. \
        The guest didn’t respond earlier, so repeat the offer with a warm, \
            brief follow-up. Greet again lightly and ask what they need help with: \
                lock door, order refills, Wi-Fi info, or call reception.\
                    Keep it modern, natural, and under 1.5 lines—no emojis, no filler." 
                
    greeting2 ="Hello Again, can you try one more time I did not get that in natural way "
    
    if state["greeting"] == "not_done":
        greeting_add = model.invoke("make a 1 - 2 line message with these instructions for " + greeting1+ state["messages"]).content
        state['llm_response'] =  greeting_add 
    
    if state["greeting"] == "done":
        greeting_add =model.invoke("you can include in the message \
            what guest spoke to us last to seem natural" + state ["messages"] +\
            + " in 1 line -  I can help you with these services lock door ,\
            give wifi informtion, or call reception for you, refills for snacks or drinks" + greeting2).content
        state['llm_response'] = greeting_add
    
    state["greeting"] = "done"
    return state



#NOW BUILD THE GRAPH
# Add nodes and edges to the graph
graph_builder.add_node("process_input", process_input)
graph_builder.add_node("which_door", which_door)
graph_builder.add_node("which_refill", which_refill)
graph_builder.add_node("reception", reception )
graph_builder.add_node("wifi_info", wifi_info )
graph_builder.add_node("ask_service_again",ask_service_again)
graph_builder.add_node("door_locked_message", door_locked_message)
graph_builder.add_node("ask_door_name", ask_door_name)
graph_builder.add_node("ask_refill_again" , ask_refill_again)
graph_builder.add_node("summary", summary)

#GRAPH EDGES AND CONDITIONAL ROUTES
graph_builder.add_edge(START, "process_input")
graph_builder.add_conditional_edges( "process_input", which_service ,
                                    {
                                      "which_door" :"which_door",
                                        "which_refill": "which_refill",
                                        "reception":"reception",
                                        "wifi_info" :"wifi_info",
                                        "ask_service_again":"ask_service_again"
                                    }
                                    )

graph_builder.add_conditional_edges("which_door", find_door ,
                                    {
                                        "door_locked_message": "door_locked_message",
                                        "ask_door_name": "ask_door_name",
                                    }
                                    )

graph_builder.add_edge("process_input", END)
graph_builder.add_edge("ask_door_name", END)
graph_builder.add_edge("reception", END)
graph_builder.add_edge("wifi_info", END)
graph_builder.add_edge("door_locked_message", END)
graph_builder.add_conditional_edges("which_refill", find_refill, {
                    "ask_refill_again" : "ask_refill_again" ,
                    "summary": "summary"
                    })
graph_builder.add_edge("summary", END)
graph_builder.add_edge("ask_refill_again" , END)
# Compile the graph
graph = graph_builder.compile()

#PLOT THE GRAPH OPTIONAL
#with open("graph_home.png", "wb") as f:f.write(graph.get_graph().draw_mermaid_png())



#STREAMLIT APP STARTS HERE
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Define the chat function
def chat(user_input):
    st.session_state.messages.append(HumanMessage(content=user_input))
    print(st.session_state.messages)
    for msg in st.session_state.messages:
        value = msg.content
    print(value , "USER VALUSE")
    output = graph.invoke({"messages": value})
    print(output, "****OUTPUT FROM GRAPH AND WHAT IS SHOWN ON UI IS LLM _RESPONSE")
    response = output["llm_response"]
    st.session_state.messages.append(AIMessage(content=response))
    return response


# Inject dark mode styles + audio player
st.markdown("""
    <style>
    /* Base layout and text color */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }

    h1, h2, h3, p, label, div {
        color: #ffffff !important;
    }

    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 6px !important;
        padding: 10px !important;
        outline: none !important;
        transition: border 0.3s ease;
    }

    /* Blue border on focus for input */
    .stTextInput>div>div>input:active {
        border: 1px solid #1e90ff !important;  /* DodgerBlue */
        box-shadow: 0 0 0 1px #1e90ff !important;
    }

    .st-b3 st-b8 st-bu st-b1 st-bl st-ae st-af st-ag st-ah st-ai st-aj st-bv st-bq:active {
        border : 1px solid #1e90ff !important;
    }
    
    .st-ba {
       border : 1px solid white!important; 
    } 
    
    /* Button styling */
    .st-emotion-cache-ocsh0s {
        width: 100% !important;
        background-color: #31323e!important;
        color: white!important;
        border: 0px solid ;
        padding: 9px,3px,9px,3px!important;
        border-radius: 6px;
        font-size: 16px;
        margin-top: 10px;
        transition: all 0.2s ease;
    }

    .st-emotion-cache-7czcpc{
        width:300px!importtant;
        height:300px!importtant;
    }
    
    .st-emotion-cache-16tyu1 h1
    {
            font-size: 2.0rem!important;
    }

    /* Blue outline on focus for button */
    .st-emotion-cache-ocsh0s:focus {
        border: 1px solid #1e90ff !important;
        box-shadow: 0 0 0 2px #1e90ff !important;
        outline: none!important;
        color : black!important;
    }
    </style>
""", unsafe_allow_html=True)


# Create a Streamlit app
with st.form(key="input_form"):
    
    st.title("Condo Guest Services")
    st.write("Langchain Graphs with Tools, Memory, Controlled Outputs")
    image_url = "/voice_agent1.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(image_url, use_container_width=True)
    user_input = st.text_input("How can I help ?", key="user_input")
    submit_button = st.form_submit_button(label='Send')



def generate_audio(text):
    audio_stream = client.text_to_speech.convert(
    text=text,
    voice_id="ELEVEN LABS ID",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",) 
    audio_bytes = io.BytesIO()
    for chunk in audio_stream:
        if chunk:
            audio_bytes.write(chunk)
    audio_bytes.seek(0)
    return audio_bytes

if submit_button:
    response = chat(user_input)
    audio_bytes = generate_audio(response)
    st.audio(audio_bytes.read(), format="audio/mp3", start_time=0, autoplay=True)
    st.write("Assistant:", response)
    
    

