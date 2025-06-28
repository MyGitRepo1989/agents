import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.types import Send
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
import random
from typing import Literal
from typing_extensions import Literal
from IPython.display import Image, display
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


llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)


#this is the multiple response 
subjects_prompt= """
generate a list of comma seperated list of between 10 to 15 \
    examples related to \
    {topic}.
"""

#final response to user
joke_prompt = """Generate the best response for encouranging sales to a customer between 1 to 2 lines maximum {subject}"""

#selects the best one
best_joke_prompt ="""below are a bunch responses to customer about {topic}. Select the best one that can encourange sales. Return the id of the best one {jokes}
"""

class Subjects(BaseModel):
    subjects: list[str]
    
class Joke(BaseModel):
    joke:str

class BestJoke(BaseModel):
    id : int = Field(description = "Index of best response , starting with 0")
    
    
model = ChatOpenAI(model ="gpt-4o-2024-08-06", temperature=0)


#setup your state here
class OverallState(TypedDict):
    topic: str
    subjects: list
       # Notice here we use the operator.add
        # This is because we want combine all the jokes we generate
        # from individual nodes back into one list - this is essentially
        # the "reduce" part
    jokes: Annotated[list, operator.add]
    best_selected_joke: str
    

class JokeState(TypedDict):
    subject: str
    
def generate_topics(state:OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])  
    # model with structured output CLASS PYDANTIC SCHEMA invoke prompt  
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects} #class subjects returns list of subjects

def generate_joke(state: JokeState):
    prompt= joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes" :[response.joke]}


#here we will define the logic to map out the generated subjects
#we will use this as an edge of the 
def continue_to_jokes(state: OverallState):
    # we will return a list of "send" objects
    # each "send" object consists of the name of a node
    return [Send("generate_joke" , {"subject": s }) for s in state ["subjects"]]

def best_joke(state: OverallState):
    jokes = "\n\n".join (state["jokes"])
    prompt=best_joke_prompt.format(topic=state["topic"],jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


#graph setup is here
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics",continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke","best_joke" )
graph.add_edge("best_joke",END )
app = graph.compile()

#extract graph 
with open("agent_topic_extract.png", "wb") as f:f.write(app.get_graph().draw_mermaid_png())

#your LLM loop goes here
user_input= input("please add a typical customer call query  : ")
[print(s) for s in app.stream({"topic": user_input})]






