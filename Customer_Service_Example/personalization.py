from typing_extensions import TypedDict
import random
from typing import Literal
from typing_extensions import Literal
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
memory = MemorySaver()

# THIS IS THE STEP WISE FUNNEL
# recommended_products
# interests - structured outputs
# personalized greeting  - saving tips or investing tips
# look up ID share balance
# invest or save in products/ discounted category  

# 1 def check the ID  if not then ask again
# 2 lets have a personalized greeting with slogan
# 2 share the balance 
# 3 share  send savings tips OR Investment Tips 
# 4 prepare a report of their profile

#SET UP THE LLM
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
llm_general = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)


#READ YOUR RESOURCE CAN BE ANYTHING IN YOUR ORGANISATION
df = pd.read_csv("data_files/customer_info.csv")
print(df.head(3))
cust_id_list= df["customer_id"].values.tolist()
df = df.fillna(0)

# SOME DATA ENGINEERING
# Add two new columns: feedback and risk_level
df['feedback'] = 'None'
df['risk_level'] = 'None'
print(df.columns)



#DEFINE YOUR STATE GRAPH
class State(TypedDict):
    # annotated  means its a list and its a list beacuse of memory
    messages: Annotated[list, add_messages]
    cust_id_list: list
    cust_id : int
    id_found: str
    summary: dict
    summary_analysis :str
    number_cc:int
    balance:int
    salary:int
    score:int
    feedback:str
    payment_amount:int
    update_balance :str
    


def find_user_id(state:State):
    """ use this to find the cust_id of user from cust_id_list"""
    print("here")
    print(state['messages'][0].content)
    state['cust_id'] = int(state['messages'][0].content)
    print(state)
    if isinstance(state['cust_id'], int) and state['cust_id'] >= 0:   
        state['cust_id'] = int(state['cust_id'])
        print("i am int")
        if state['cust_id'] in df["customer_id"].values:
            state['id_found']= "found" 
            print(state['id_found'])
            return state
        else:  
            state['id_found']= "not_found"
            print("i am not int",state['id_found'], state['cust_id'] )
            print(state)
            return state
    else:
        state['id_found']= "not_found"
        return state



def chatbot(state: State):
    ("THIS IS THE MAIN chatbot")
    if state.get('id_found') == "not_found":
        #state["messages"].append({"role": "assistant", "content": "pls enter a numeric value and correct number"})
        return state
    elif state.get('id_found') == "found":
        #state["messages"].append({"role": "assistant", "content": "I found your order : we are getting your information pls wait"})
        return state
    else:
        # Handle any other cases
        return state


def route_after_id(state: State):
    print("WE ARE FINDING CUSTOMER ID IN THE GRAPH" , state)
    route_id= state.get('id_found')
    print(route_id)
    if route_id == "found":
        return "found"
    if route_id == "not_found":
        return "not_found"
  

def get_summary(state:State):
    print("SUMMARY OF CUSTOMER : ", state)
    summary_df = df.loc[df['customer_id']==state['cust_id']]
    summary_df = summary_df.fillna(0).replace(to_replace=['na', 'NA', 'Na', 'nA'], value=0)
    state["summary"]= summary_df.to_dict(orient='records')[0]
    print(state["summary"],summary_df)
    
    result = llm_general.invoke(f'Create an organised way of client information, {state["summary"]}, \
        then summarize this information in 2-3 lines starting with: Summary of your Account ID {state["cust_id"]}.')
    state["summary_analysis"] = result.content
    print(state["summary_analysis"])
    
    state["number_cc"]=int(df['credit_card'].loc[df['customer_id']==state['cust_id']])
    state["balance"]=int(df['balance'].loc[df['customer_id']==state['cust_id']])
    state["salary"]=int(df['estimated_salary'].loc[df['customer_id']==state['cust_id']])
    state["score"]=int(df['credit_score'].loc[df['customer_id']==state['cust_id']])
    print(state["number_cc"],"nocard", state["balance"] ,"balance", state["salary"], "salary", state["score"],"score")
    
    return state


def client_question_router(state:State):
    print("FUNNEL AND ROUTING CUSTOMER PROFILE", state)
    if state["number_cc"] >=1:
        return "send_balance" 
    if state["salary"] >= df['estimated_salary'].mean():
        return "investment_strategy"
    if state["score"] <= df['estimated_salary'].mean():
        return "mark_as_risk"
  

def send_balance(state:State):
    print("PRINTING CUSTOMER BALANCE")
    print(state)
    print(llm)
    print(state["number_cc"],"nocard", state["balance"] ,"balance", state["salary"], "salary", state["score"],"score")
    result = llm_general.invoke(f"Please inform customer that for this Customer ID {state['cust_id']}, \
        they have number of Credit Cards {state['number_cc']}, of existing balance {state['balance']}")
    balance_summary = result.content
    print(balance_summary)
    print("EXIT NODE OF BALANCE")
  
 
    #ask user if they want to update balance make a payment
    new_update_balance ="non"
    new_update_balance = input(str("Do you want to make a payment? Yes / No: "))
    if str(new_update_balance.lower()) == "yes":
        state["update_balance"] = "Yes"
    if str(new_update_balance.lower()) == "no":
        state["update_balance"] = "No"
    else:
        state["update_balance"] = "Yes"
    print(state["update_balance"])         
    print("leaving yes no " )
    return state
   

  
def make_payment_router(state:State):
    print("MAKE A PAYMENT NODE FOR USER ")
    print(state)
    print("SPECIFIC STATE", state["update_balance"])
    if state["update_balance"] == "Yes":
        return "Yes"
    if state["update_balance"] == "No":
        return "No"


def make_payment(state:State):
    print("NODE THAT WILL MAKE A PAYMENT")
    while True:
        payment_amount = int(input("Please add payment amount: "))
        if payment_amount > 0:
            break
        else:
            payment_amount = int(input("Please add payment amount: "))

    df.loc[df["customer_id"] == state["cust_id"], "balance"] -= payment_amount
    new_balance = df.loc[df["customer_id"] == state["cust_id"], "balance"].values[0]

    print("I made your payment for: ", str(payment_amount), "your new balance is: ", str(new_balance))
    return state


def ask_order_id(state: State):
    print("HELLO TRY AGAIN", state)
    while state['id_found'] == "not_found":
        try:
            new_input = int(input("please share ur ID eg:15647311 "))
            print(new_input)
            state["cust_id"] = int(new_input)
            print("UPDATED:", state)
            if state['cust_id'] in df["customer_id"].values:
                state['id_found']= "found"
            else:
                state['id_found']= "not_found"   
        except ValueError:
            print("Invalid input! Please enter a valid integer ID.")
    return state


def investment_strategy(state:State):
    normal_salary = state["salary"] >= df['estimated_salary'].mean()
    result = llm_general.invoke(
        f"Send a message for this customer saying Congrats you qualify for investment services, "
        f"their salary is healthy and above {normal_salary}, "
        f"with {state['summary_analysis']} they should contact Investment Services at 18009876513"
    )
    print(result.content)
    return state


def mark_as_risk(state:State):
    df.loc[df["customer_id"] == state["cust_id"], "risk_level"] -= "High_Risk"
    print("I marked the user as high risk")
    return state


#FINALLY TAKE FEEDBACK
def leavefeedback(state:State):
    user_feedback = str(input("Before you leave do you want to leave us feedback?: "))
    df.loc[df["customer_id"] == state["cust_id"], "feedback"] = str(user_feedback)
    print(f'I added your feedback Thank you have a nice day {user_feedback}')
    return state


#GRAPH BUILDER
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("find_user_id", find_user_id)
graph_builder.add_node("get_summary", get_summary)
graph_builder.add_node("ask_order_id", ask_order_id)
graph_builder.add_node("investment_strategy", investment_strategy)
graph_builder.add_node("send_balance",  send_balance)
graph_builder.add_node("mark_as_risk", mark_as_risk)
graph_builder.add_node("make_payment",make_payment)
graph_builder.add_node("leavefeedback",leavefeedback)


#GRAPH EDGES AND CONDITIONS
graph_builder.add_edge(START, "find_user_id")
graph_builder.add_edge("find_user_id", "chatbot") 
graph_builder.add_conditional_edges("chatbot", route_after_id,
{
    "not_found" : "ask_order_id",
    "found" : "get_summary"
})


graph_builder.add_conditional_edges("get_summary", client_question_router,
{
    "send_balance" : "send_balance",
    "investment_strategy" : "investment_strategy",
    "mark_as_risk" : "mark_as_risk",
})


graph_builder.add_conditional_edges("send_balance", make_payment_router,
{
    "Yes" :"make_payment",
    "No" : "investment_strategy",
})

graph_builder.add_edge("ask_order_id","chatbot") 
graph_builder.add_edge("make_payment","leavefeedback")
graph_builder.add_edge("investment_strategy","leavefeedback")
graph_builder.add_edge("mark_as_risk","leavefeedback")
graph_builder.add_edge("leavefeedback", END)

config = {"configurable": {"thread_id": "1"}}
graph = graph_builder.compile(checkpointer=memory)




#TEST THE GRAPH BELOW
user_in_message = int(input("please share ur  ID eg:15647311 "))
ai_msg = graph.invoke({"messages": [{"role": "user", "content": str(user_in_message)}]}, config=config)
print(ai_msg)
 
#with open("graph.png", "wb") as f:f.write(graph.get_graph().draw_mermaid_png())
#open("graph2.png", "wb").write(graph.get_graph().draw_mermaid_png())