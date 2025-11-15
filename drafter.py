from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()  # take environment variables from .env file

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tools
@tool
def update(content: str) -> str:
    """ Update the document with new content. """
    global document_content
    document_content = content
    return "Document updated."

@tool
def save(filename: str) -> str:
    """ Save the document to a text file and finish the process.

        Args:
            filename (str): The name of the file to save the document to.
       """
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"
    
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Document saved to {filename}.")
        return f"Document saved to {filename}."
    except Exception as e:
        return f"Failed to save document: {str(e)}"
    

tools = [update, save]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""You are a helpful assistant, help the user to edit and save their document using the provided tools. Return the final result as 'Result':[result] when saving is done.
                                  the current document content is: {document_content}
                                  """)
    
    if not state["messages"]:
        user_input = "I am ready to help you update a document, what would you like to create?"
        user_message = HumanMessage(role="user", content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"User input: {user_input}")
        user_message = HumanMessage(role="user", content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """ Determine if we should continue the agent loop. """
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app = graph.compile()

from IPython.display import display, Image
display(Image(app.get_graph().draw_mermaid_png()))

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()