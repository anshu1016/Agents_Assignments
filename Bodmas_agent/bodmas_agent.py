# from dotenv import load_dotenv
# import os
# load_dotenv()
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.tools import tool
# from langchain_groq import ChatGroq
# from langgraph.graph.message import add_messages
# from typing import Annotated
# from langchain_core.messages import AnyMessage
# from typing_extensions import TypedDict
# from langchain_core.messages import HumanMessage,SystemMessage
# from langgraph.graph import START,END,StateGraph
# from langgraph.prebuilt import tools_condition, ToolNode
# from IPython.display import Image,display


# API_KEYS = ['GROQ_API_KEY','GROQ_YOUTUBE_COLLAB','HUGGINGFACE_TOKEN','LANGCHAIN_API_KEY','LANGCHAIN_PROJECT','HF_TOKEN','GOOGLE_API_KEY']
# for k in API_KEYS:
#   value = os.getenv(k)
#   if k is not None:
#     keys = f'{k} : {os.getenv(k)}'
#     os.environ[k] = value
#     print(keys)
#   else:
#         print(f"{k} not found in userdata")



# from typing import Union
# from langchain_groq import ChatGroq
# def multiply(a: Union[int, str], b: Union[int, str]) -> int:
#     '''
#     Multiply a and b.

#     Args:
#         a: First number (int or str representing int)
#         b: Second number (int or str representing int)

#     Returns:
#         Product as an integer.
#     '''
#     return int(a) * int(b)


# def add(a: Union[int, str], b: Union[int, str]) -> int:
#     '''
#     Add a and b.

#     Args:
#         a: First number (int or str representing int)
#         b: Second number (int or str representing int)

#     Returns:
#         Sum as an integer.
#     '''
#     return int(a) + int(b)


# def divide(a: Union[int, str], b: Union[int, str]) -> int:
#     '''
#     Divide a by b.

#     Args:
#         a: Numerator (int or str representing int)
#         b: Denominator (int or str representing int)

#     Returns:
#         Quotient as a float.

#     Raises:
#         ValueError: If b is zero.
#     '''
#     a, b = int(a), int(b)
#     if b == 0:
#         raise ValueError("Division by zero is not allowed.")
#     return a / b


# def subtract(a: Union[int, str], b: Union[int, str]) -> int:
#     '''
#     Subtract b from a.

#     Args:
#         a: First number (int or str representing int)
#         b: Second number (int or str representing int)

#     Returns:
#         Difference as an integer.
#     '''
#     return int(a) - int(b)

# def make_bodmas_graph():
#     @tool
#     def multiply(a: Union[int, str], b: Union[int, str]) -> int:
#         '''
#         Multiply a and b.

#         Args:
#             a: First number (int or str representing int)
#             b: Second number (int or str representing int)

#         Returns:
#             Product as an integer.
#         '''
#         return int(a) * int(b)

# @tool
# def add(a: Union[int, str], b: Union[int, str]) -> int:
#         '''
#         Add a and b.

#         Args:
#             a: First number (int or str representing int)
#             b: Second number (int or str representing int)

#         Returns:
#             Sum as an integer.
#         '''
#         return int(a) + int(b)

# @tool
#     def divide(a: Union[int, str], b: Union[int, str]) -> int:
#         '''
#         Divide a by b.

#         Args:
#             a: Numerator (int or str representing int)
#             b: Denominator (int or str representing int)

#         Returns:
#             Quotient as a float.

#         Raises:
#             ValueError: If b is zero.
#         '''
#         a, b = int(a), int(b)
#         if b == 0:
#             raise ValueError("Division by zero is not allowed.")
#         return a / b

# @tool
#     def subtract(a: Union[int, str], b: Union[int, str]) -> int:
#         '''
#         Subtract b from a.

#         Args:
#             a: First number (int or str representing int)
#             b: Second number (int or str representing int)

#         Returns:
#             Difference as an integer.
#         '''
#         return int(a) - int(b)

#         tools=[add,multiply,divide,subtract]

#         llm = ChatGroq(model='Llama3-8b-8192')
#         llm_with_tools = llm.bind_tools(tools,parallel_tool_calls = False)


#     class MessagesState(TypedDict):
#         messages: Annotated[list[AnyMessage],add_messages]


#     sys_msg = SystemMessage(content = 'You are a helpful assistant tasked with performing arithmetic set of inputs.You know well about BODMAS techniques.')

#     def assistant(state: MessagesState):
#         return {"messages":[llm_with_tools.invoke([sys_msg]+state['messages'])]}


#     builder = StateGraph(MessagesState)

#     builder.add_node('assostant',assistant)
#     builder.add_node('tools',ToolNode(tools))

#     builder.add_edge(START,'assistant')
#     builder.add_conditional_edges('assistant',tools_condition)
#     builder.add_edge('tools','assistant')
#     builder.add_edge('assistant',END)


#     agent = builder.compile()
#     return agent



from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, Union
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
# Load and print API keys
API_KEYS = ['GROQ_API_KEY', 'GROQ_YOUTUBE_COLLAB', 'HUGGINGFACE_TOKEN', 'LANGCHAIN_API_KEY', 'LANGCHAIN_PROJECT', 'HF_TOKEN', 'GOOGLE_API_KEY']
for k in API_KEYS:
    value = os.getenv(k)
    if value is not None:
        keys = f'{k} : {value}'
        os.environ[k] = value
        print(keys)
    else:
        print(f"{k} not found in userdata")


# Tools with @tool decorator
@tool
def multiply(a: Union[int, str], b: Union[int, str]) -> int:
    '''
    Multiply a and b.

    Args:
        a: First number (int or str representing int)
        b: Second number (int or str representing int)

    Returns:
        Product as an integer.
    '''
    return int(a) * int(b)

@tool
def add(a: Union[int, str], b: Union[int, str]) -> int:
    '''
    Add a and b.

    Args:
        a: First number (int or str representing int)
        b: Second number (int or str representing int)

    Returns:
        Sum as an integer.
    '''
    return int(a) + int(b)

@tool
def divide(a: Union[int, str,float], b: Union[int, str,float]) -> int:
    '''
    Divide a by b.

    Args:
        a: Numerator (int or str representing int)
        b: Denominator (int or str representing int)

    Returns:
        Quotient as a Int.

    Raises:
        ValueError: If b is zero.
    '''
    a, b = int(a), int(b)
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@tool
def subtract(a: Union[int, str], b: Union[int, str]) -> int:
    '''
    Subtract b from a.

    Args:
        a: First number (int or str representing int)
        b: Second number (int or str representing int)

    Returns:
        Difference as an integer.
    '''
    return int(a) - int(b)



def make_bodmas_graph():
    tools = [add, multiply, divide, subtract]

    llm = ChatGroq(model='Llama3-8b-8192')
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    sys_msg = SystemMessage(content='You are a helpful assistant tasked with performing arithmetic set of inputs. You know well about BODMAS techniques.')

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}
    
    def should_continue(state: MessagesState):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    builder = StateGraph(MessagesState)

    builder.add_node('assistant', assistant)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'assistant')
    builder.add_conditional_edges('assistant', tools_condition)
    builder.add_conditional_edges('assistant', should_continue)
    builder.add_edge('tools', 'assistant')
    # builder.add_edge('assistant', END)

    agent = builder.compile()
    return agent


agent = make_bodmas_graph()