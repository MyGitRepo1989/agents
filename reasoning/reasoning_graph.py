from typing_extensions import TypedDict
import random
from typing import Literal
from typing_extensions import Literal
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from openai import OpenAI
import re
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
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


#********REASONING SET UP********************************************** 

llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
reasoning_client = OpenAI()

reasoning_prompt = " Instructions:\
- You will read the context of the world \
- The context of the world will help you understand how certain topics impact economy, markets , stocks  \
- You will use The context of the world when you reason \
- you will next read the news headlines of the day , these are the current affairs of the world\
- using the context of world you will reason what impact the headlines will have on economy\
- you will then as final step of your reason tell why the headlines have neutral, low positive, positive , low negative , negative , highly positive or highly negative impact on stock market\
- knowing how markets, stock markets and economy reacts to current affairs give your final reason\
- in the last step you will provide over all conclusion with final impact of all the headlines  \
- if you dont know say I dont know"



news_file_loaction = "news_data/news_2025-06-20.csv"
news_df = pd.read_csv(news_file_loaction)
print("THE NEWS FILE WE ARE ANALYSISNG IS :",news_df.head(3))


#******************************************************

# THESE ARE THE STRUCTURED OUTPUT DICT

# Schema for structured output to use as routing logic
class Route_level(BaseModel):
    step: Literal[  
    "MajorCrisis",
    "MarketSector",
    "Geopolitical",
    "Company",
    "Government",
    "Other"] = Field(
        None, description="The next step in the routing process"
    )

# Schema for structured output to use as routing logic
class Route_Impact(BaseModel):
    step: Literal[  
    "LowPositive",
    "LowNegative",
    "Positive",
    "Negative",
    "HighPositive",
    "HighNegative",
    "Neutral",
    ] = Field(
        None, description="The next step in the routing process"
    )

 
# Schema for structured output to use as routing logic
class Route_type(BaseModel):
    step: Literal[
    "Politics",
    "Finance",
    "Economy",
    "Business",
    "Law&Regulations",
    "Technology&Science",
    "LifeStyle",
    "Health&Medicine",
    "Sports",
    "Art&Entertainment",
    "Governance",
    "Other"] = Field(
        None, description="The next step in the routing process"
    )
    

#******************************************************

# ROUTING LLM
    
# Augment the LLM with schema for structured output
router_type = llm.with_structured_output(Route_type)
router_level = llm.with_structured_output(Route_level)
router_impact = llm.with_structured_output(Route_Impact)


#GRAPH 1 creating content groups 
#***************************************************************
# NOW WE ARE DEFINING THE GRAPH FOR TYPES 
   
class State(TypedDict, total=False):
    headline: str
    decision_type: str
    decision_level: str
    index_news: int
    current_index: int
    headlineinde : int
    decision_economy :str
   

def getheadline(state:State):
    state["headline"] = str(news_df["Headline"][state["current_index"]])+str(news_df['Description'][state["current_index"]])+str(news_df['Content'][state["current_index"]])
    print("NEWS TITLE : ",state["headline"])
    return state

def get_type(state:State):
    type_decision = router_type.invoke(f'You will analyse what category type this news headline {state["headline"]} is of\
    Politics - if its geopolitical or politics news, \
    Finance - if its financial news, \
    Economy - if it impacts economy, \
    Business- if its company or market related, \
    Law&Regulations - if its law, crime, justice, regulation news, \
    Technology&Science - it its technology or science related, \
    LifeStyle - if its just about travel etc news, \
    Health&Medicine - anything with health , medicine mews, \
    Sports - sports news, \
    Art&Entertainment - arts, movies, entertainment, television music etc, \
    Governance - any governance related news, \
    If it is anything else return Other')
    
    print("TYPE DECISION",type_decision.step)
    state["decision_type"]= type_decision.step
    return state



#******************************************************
# GRAPH 1
news_graph = StateGraph(State)
news_graph.add_node("getheadline",getheadline)
news_graph.add_node("get_type",get_type)


news_graph.set_entry_point("getheadline")
news_graph.add_edge("getheadline", "get_type")
news_graph.add_edge("get_type", END) 
type_app = news_graph.compile()


#SAVE A TEMP DF
no_headlines= news_df.shape[0]
# Initialize an empty list to store results
results = []

# Loop through the first 5 headlines
for i in range(no_headlines):
    # Get output from type_app
    output = type_app.invoke({"current_index": i})
  
    # Extract decision type and level
    decision_type = output["decision_type"]

    
    # Append results as a dictionary to the list
    results.append({
        "Headline": str(news_df["Headline"][i]) + " " + str(news_df["Description"][i]),
        "Type": decision_type,
    })
    
    # Print results (optional, for debugging)
    print( news_df["Headline"][i], decision_type )

# Create a DataFrame from the results list
result_df = pd.DataFrame(results)

# Display the DataFrame
print("TYPES FROM THE NEWS ARE EXTRACTED : ",result_df)
print(pd.Series(result_df["Type"].value_counts()))
news_impact_for_reasoning = "news_data/data_reasoning.csv"
result_df.to_csv(news_impact_for_reasoning, index=False)





#PART TWO REASONING GRAPH
#****************************************************

#lOAD CSV and data
knowledge_csv= pd.read_csv("data_files/world_corpus.csv")
knowledge_csv.columns=["topic","context"]
print("THIS IS THE KNOWLEDGE CSV",knowledge_csv.head(2))


#LOAD THE TYPES FROM PART ONE  OF THE NEWS 
result_df = pd.read_csv(news_impact_for_reasoning)
catogories = pd.Series(result_df["Type"].value_counts()).index.to_list()
print("CATEGORIES OF NEWS I WILL USE ", catogories)



class KnowledgeDict(TypedDict):
    topic: str
    value: str

class WorldState(TypedDict, total=False):
    headline:str
    knowledge_info:dict[str, str]
    type_corpus:str
    reasoning_result:str
    type_cat :str 
    impact_decision:str

world_graph = StateGraph(WorldState)

def get_world_update(state:WorldState):
    print("Node for getting knowledge of world")
    print(knowledge_csv)
    knowledge_info = knowledge_csv.set_index("topic")["context"].to_dict()
    print(knowledge_info)
    state["knowledge_info"] = knowledge_info 
    print("Created Dict of knowledge of world: Dict ",knowledge_info)
    return state

def get_type_headlines(state:WorldState):  
    print("Getting world information")
    type_cat = state["type_cat"]
    print(result_df["Headline"].loc[result_df["Type"]==type_cat])
    print(result_df["Headline"].loc[result_df["Type"]==type_cat].shape)
    print("********")
    
    type_corpus = " ".join((result_df["Headline"]).loc[result_df["Type"]==type_cat].tolist())
    print("Created groups of information", type_corpus)
    state["type_corpus"] = type_corpus
    return state


def reasoning_model(state:WorldState):
    print("REASONING")
    print("\n" * 4)
    world_dict = state["knowledge_info"]
    corpus = state["type_corpus"]
    context = f'{reasoning_prompt} corpus of news headlines {corpus} context of the world {world_dict}'

    response = reasoning_client.responses.create(
        model="o4-mini",
        reasoning={"effort": "high",    
                },
        input=[
            {
                "role": "user", 
                "content": context
            }
        ],
        #max_output_tokens=900,
    )
    reasoning_result = response.output_text
    state['reasoning_result'] = reasoning_result
    print(reasoning_result)
    return state

def structured_impact(state:WorldState):
    print("I AM IN STRUCTURED OUTPUT FOR REASONING")
    impact_decision = router_impact.invoke(f'Instructions :\
        - you will read the analysis of reasoning model {state["reasoning_result"]}\
        - this is the analysis if the current news has a positive , negative , neutral impact on stocks and economy\
        - you will synthesize this to give a final verdit between these options Low Positive, Low Negative\
        - Negative, Positive, Hight Positive, High Negative')
    
    print("IMPACT DECISION",impact_decision.step)
    state["impact_decision"]= impact_decision.step
    print("LEAVING STRUCTURED")
    
    #RELEASE MEMORY IN PRODUCTION
    #state["headline"] = " "
    #state["knowledge_info"] = {"a": "b" }
    #state["reasoning_result"] =" "
    #state["type_corpus"] = " "
   
    return state
   
worldview_graph = StateGraph(WorldState)

worldview_graph.add_node("get_world_update",get_world_update)
worldview_graph.add_node("get_type_headlines",get_type_headlines)
worldview_graph.add_node("reasoning_model",reasoning_model)
worldview_graph.add_node("structured_impact",structured_impact)

worldview_graph.set_entry_point("get_world_update")
worldview_graph.add_edge("get_world_update", "get_type_headlines")
worldview_graph.add_edge("get_type_headlines", "reasoning_model")
worldview_graph.add_edge("reasoning_model","structured_impact")


world_app = worldview_graph.compile()

#world_app.invoke({"type": "Politics"})   

impacts_results = []
headlines = []
reasoning_results = []
categories = []   
"""
for i in range(len(catogories)):
    type_cat = catogories[i]
    print(i ,"TYPE", type_cat )
    final_impact = world_app.invoke({"type_cat": type_cat}) 
    print(final_impact ,"\n\n\n", "************")
    impacts_results.append(final_impact["impact_decision"])
    
"""

for i in range(len(catogories)):
    type_cat = catogories[i]
    print(i ,"TYPE", type_cat )
    final_impact = world_app.invoke({"type_cat": type_cat}) 
    print(final_impact ,"\n\n\n", "************")
    impacts_results.append(final_impact["impact_decision"])
    headlines.append(final_impact["type_corpus"])
    reasoning_results.append(final_impact["reasoning_result"])
    categories.append(type_cat)

df = pd.DataFrame({
    "Category": categories,
    "Headline": headlines,
    "Reasoning Result": reasoning_results,
    "Impact Result": impacts_results
})

print(df)

print(impacts_results)
print(catogories)


#*******************************
# SAVING THE RESULT
print("I AM SAVING REASONING RESULTS")

file_name = news_file_loaction
date_extracted = re.search(r'\d{4}-\d{2}-\d{2}', file_name).group()
print(date_extracted)

# Create a new DataFrame
reasoning_impact_df = pd.DataFrame({
    'Category': catogories,
    'Impact Decision': impacts_results
})

# Save the DataFrame as CSV
new_file_name = f"news_data/reasoning_output_{date_extracted}.csv"
reasoning_impact_df.to_csv(new_file_name, index=False)



#TEST OUTPUT
"""
for i in range(len(catogories)):
    corpus_df = result_df["Headline"].loc[result_df["Type"]==catogories[i]]
    corpus = " ".join(corpus_df)
    print(corpus)
    
"""  
