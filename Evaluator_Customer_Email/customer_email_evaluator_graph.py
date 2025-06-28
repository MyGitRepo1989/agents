# Using structured output class we create a loop of a llm that does an 1- action and an 2- evaluator that checks 
# - User inputs a raw unformated text for a work email
# - the llm write a basic email
# - evaluator checks if its professional
# -  return the final proof checked email



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
import os
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

#LLM setup (assuming your OPENAI_API_KEY is set)
llm = ChatOpenAI(model="gpt-4o", temperature=0)


#THIS IS THE STATE GRAPH
# Graph state
class State(TypedDict):
    email: str  # The generated email
    original_email: str  # Input from user
    feedback: str  # Evaluator feedback
    professional_or_not: str  # Evaluation result


# Schema for structured output
class Feedback(BaseModel):
    grade: Literal["professional", "not professional"] = Field(
        description="Decide if the email is professional or not.",
    )
    feedback: str = Field(
        description="If the email is not professional, provide feedback on how to improve it.",
    )

# Augment LLM for evaluation
evaluator = llm.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
    """LLM generates a professional email"""
    if state.get("feedback"):
        msg = llm.invoke(
            f"Rewrite this email from tech lead in agile software development enviorment and is a project lead based on feedback: {state['original_email']}. Feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a professional email: {state['original_email']}")
    return {"email": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the email"""
    grade = evaluator.invoke(f"Evaluate if this email is professionally written : {state['email']}")
    return {"professional_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function
def route_email(state: State):
    """Route back to generator or end based on evaluation"""
    if state["professional_or_not"] == "professional":
        return "Accepted"
    elif state["professional_or_not"] == "not professional":
        return "Rejected + Feedback"

# Build workflow
optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_email,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"},
)
# Compile workflow
optimizer_workflow = optimizer_builder.compile()


# Save image instead (for VSCode)
from PIL import Image
with open("Email_Evaluator.png", "wb") as f:
    f.write(optimizer_workflow.get_graph().draw_mermaid_png())

#THIS IS THE TET LOOP YOU CAN ADD THIS IN YOUR CHATBOT
# Invoke with example email
while True:
    original_email  = input(" Add your email key points here and we will finish it : ")
    #original_email = "Hey dude, can u send me the stuff ASAP? Thx!"
    state = optimizer_workflow.invoke({"original_email": original_email})
    print(state["email"])