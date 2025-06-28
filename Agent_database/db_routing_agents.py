#THIS PROGRAM ILLUSTRATES ROUTING A CUSTOMER SERVICE QUERY 
# 1 - look up order number
# 2 - send to router with user message for 3 routes - status , change delivery instructions , cancel order
# 3 - each sub process has llm agents to do action and update the dataframe
# 4 - lastly we compile all off this to send a response back to user 

from PIL import Image  
from typing_extensions import TypedDict
import random
from typing import Literal
from typing_extensions import Literal
import pandas as pd
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


# Dataframe that we will update
df = pd.read_csv("data_files/orders_jan.csv")
print(df.columns)
print(df)
order_numbers = set(df['order_no'])

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["status", "delivery", "cancel"] = Field(
        None, description="The next step in the routing process"
    )

# Augment the LLM with schema for structured output
llm = ChatOpenAI(model="gpt-4o", temperature=0)
router = llm.with_structured_output(Route)


class State(TypedDict):
    input: str
    decision: str
    output: str
    order_no :int
    delivery :str
    cancel_avaliable : str
    status: str
    

def check_df(user_number,state: State):
    state['order_no'] = user_number
    state["status"] = df["status"][df['order_no'] == state['order_no']]
    state["cancel_avaliable"] = df["cancel_avaliable"][df['order_no'] == state['order_no']]
    state["delivery"]= df["delivery"][df['order_no'] == state['order_no']]
    return  state["status"] , state["cancel_avaliable"] ,state["delivery"] ,state['order_no']


# Nodes
def agent1(state: State):
    """check status of order number """  
    state["status"] , state["cancel_avaliable"] ,state["delivery"],state['order_no'] = check_df(state["order_no"],state)
    result = llm.invoke(f'rmessage reply to customer in chat that order status is {state["status"]}')
    return {"output": result.content}


def agent2(state: State):
    """update delivery instructions in the df with user input """
    state["status"] , state["cancel_avaliable"] ,state["delivery"],state['order_no'] = check_df(state["order_no"],state) 
    df.loc[(df['order_no'] == state['order_no']), 'delivery'] = state["input"]
    result = llm.invoke(f'message reply in chat to user that your request for delivery update {state["input"]} \
        for order no :{state["order_no"]} had been updated with new delivery instructions \
            {state["delivery"]}')
    
    return {"output": result.content}


def agent3(state: State):
    """User is asking of cancel is avaliable """  
    state["status"] , state["cancel_avaliable"] ,state["delivery"],state['order_no'] = check_df(state["order_no"],state)
    result = llm.invoke(f" message a replay in chat that their order cancel avaliable status  \
        {state['cancel_avaliable']} \
        include in message if the cancel_avaliable is True that they have the option to contacting agent ")
    return {"output": result.content}

def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    
    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to status, delivery, or cancel on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "status":
        return "agent1_db"
    elif state["decision"] == "delivery":
        return "agent2_status"
    elif state["decision"] == "cancel":
        return "agent3_email"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("agent1_db", agent1)
router_builder.add_node("agent2_status", agent2)
router_builder.add_node("agent3_email", agent3)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "agent1_db": "agent1_db",
        "agent2_status": "agent2_status",
        "agent3_email": "agent3_email",
    },
)
router_builder.add_edge("agent1_db", END)
router_builder.add_edge("agent2_status", END)
router_builder.add_edge("agent3_email", END)

# Compile workflow
router_workflow = router_builder.compile()

# Invoke and save image
while True:
    with open("graph.png", "wb") as f:f.write(router_workflow.get_graph().draw_mermaid_png())

    order_input = input("Please enter your order number (or 'exit' to quit): ")
    if order_input.lower() == "exit":
        break
    try:
        order_no = int(order_input)
        # Fast set lookup
        if order_no in order_numbers:
            user_query = input("What do you want help with ? \n eg:  - check status , cancel order or leave delivery instructions: ")
            if user_query.lower() == "exit":
                break
            state = router_workflow.invoke({"input": f"{user_query}", "order_no": order_no})
            print(state["output"])
        else:
           order_input = input("Please enter a vaild order number (or 'exit' to quit): ")
    except ValueError:
        order_input = input("Please enter a vaild order number (or 'exit' to quit): ")
        

