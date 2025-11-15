from tool import tools
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import json
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START, MessagesState

load_dotenv()

# Track tool calls
tool_calls_made = []

# Debug: Print available tools
print("=" * 70)
print("📅 GOOGLE CALENDAR ASSISTANT")
print("=" * 70)
print("\n🔧 Available tools:")
for tool in tools:
    print(f"   • {tool.name}: {tool.description}")
print()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    
    # Add system message to guide the model to use tools
    system_prompt = SystemMessage(content="""You are a helpful calendar assistant. When users ask you to create events, 
    add events, or schedule something, you should use the create_calendar_event tool to do so. 
    Extract the event details from the user's message and use the create_calendar_event tool with the appropriate parameters.
    Required parameters: summary (event title), start_datetime (format: 'YYYY-MM-DD HH:MM:SS'), end_datetime (format: 'YYYY-MM-DD HH:MM:SS').
    Optional parameters: timezone (default: 'UTC'), location, description, reminders.
    If the user doesn't specify a timezone, use 'UTC' or infer from context (e.g., 'Asia/Kolkata' for India).
    For dates like "11/11/2025 at 11:00 AM", parse them into the required format and create the event.""")
    
    all_messages = [system_prompt] + messages
    response = model.invoke(all_messages)
    
    # Track tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_calls_made.append({
                'name': tool_call.get('name', 'unknown'),
                'args': tool_call.get('args', {}),
                'id': tool_call.get('id', 'unknown')
            })
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


workflow = StateGraph(MessagesState)


workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "model")

workflow.add_conditional_edges(
    # First, we define the start node. We use `model`.
    # This means these are the edges taken after the `model` node is called.
    "model",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
workflow.add_edge("tools", 'model')

app = workflow.compile()

# Use HumanMessage instead of dict format
print("\n" + "=" * 70)
print("🚀 PROCESSING REQUEST")
print("=" * 70)
user_input = "add birthday for Salma in my calendar on 11/11/2025 at 3:00 AM"
print(f"\n👤 User: {user_input}\n")

final_state = app.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config={"configurable": {"thread_id": 42}}
)

# Beautiful output formatting
print("\n" + "=" * 70)
print("📊 EXECUTION SUMMARY")
print("=" * 70)

# Show tool calls
if tool_calls_made:
    print(f"\n✅ Tools Called: {len(tool_calls_made)}")
    for i, tool_call in enumerate(tool_calls_made, 1):
        print(f"\n   [{i}] 🔧 {tool_call['name']}")
        print(f"       Arguments:")
        for key, value in tool_call['args'].items():
            # Format the value nicely
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            print(f"         • {key}: {value}")
else:
    print("\n❌ No tools were called")

# Show all messages in the conversation
print(f"\n💬 Conversation Flow ({len(final_state['messages'])} messages):")
print("-" * 70)
for i, msg in enumerate(final_state['messages'], 1):
    msg_type = type(msg).__name__
    if isinstance(msg, HumanMessage):
        print(f"\n[{i}] 👤 Human: {msg.content}")
    elif isinstance(msg, AIMessage):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\n[{i}] 🤖 AI (with tool calls):")
            for tc in msg.tool_calls:
                print(f"     🔧 Tool: {tc.get('name', 'unknown')}")
        else:
            print(f"\n[{i}] 🤖 AI: {msg.content}")
    elif isinstance(msg, ToolMessage):
        content = msg.content
        # Try to parse JSON if it's a JSON string
        try:
            if isinstance(content, str):
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2)
        except:
            pass
        print(f"\n[{i}] 🔧 Tool Result: {content}")
    else:
        # Other message types
        content = getattr(msg, 'content', str(msg))
        print(f"\n[{i}] 📨 {msg_type}: {content}")

# Final result
print("\n" + "=" * 70)
print("✨ FINAL RESULT")
print("=" * 70)
result = final_state["messages"][-1].content
print(f"\n{result}\n")
print("=" * 70)