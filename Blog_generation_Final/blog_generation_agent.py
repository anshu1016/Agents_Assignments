from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()


import os

API_KEYS = ['GROQ_API_KEY','GROQ_YOUTUBE_COLLAB','HUGGINGFACE_TOKEN','LANGCHAIN_API_KEY','LANGCHAIN_PROJECT','HF_TOKEN','GOOGLE_API_KEY']
for k in API_KEYS:
  value = os.getenv(k)
  if k is not None:
    keys = f'{k} : {os.getenv(k)}'
    os.environ[k] = value
    print(keys)
  else:
        print(f"{k} not found in userdata")


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from IPython.display import Image,display
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def create_blog_agent():
    builder2 = StateGraph(State)
    from langchain_groq import ChatGroq
    llm = ChatGroq(
            model='Llama3-8b-8192',
        )
    
    def generate_blog_agent(state: dict) -> dict:
        '''
        state: Contains current state of the graph (including 'messages').
        Returns the blog content as a dict to be stored in state.
        '''
        blog_prompt = state["messages"][-1].content  # Get the last message (user prompt)
        
        # Create a message with your system prompt
        system_msg = SystemMessage(content=f'''
        You are an expert content creator with deep knowledge of LinkedIn content strategy, personal branding, and audience psychology.

        Your task is to deeply analyze the given topic: "{blog_prompt}" and generate a powerful, insight-driven LinkedIn-style blog post of approximately 300 words.

        ✍️ Follow this structure:

        1. Hook (1–2 lines): Start with a bold or relatable statement/question to instantly grab attention.

        2. Context (2–3 lines): Introduce the topic and why it matters today — trends, pain points, or relevance.

        3. Deep Dive (4–5 lines): Break down key concepts, benefits, challenges, or misconceptions around the topic.

        4. Insight/Takeaway (3–4 lines): Offer your unique perspective, actionable advice, or lesson learned.

        5. Call to Engage (1 line): End with a question, reflection, or CTA that encourages comments, reposts, or DMs.
        ''')

        result = llm.invoke([system_msg])  # Use LLM to get result

        return {
            "messages": [result],
            "blog_content": result.content  # Optional, but helpful for reuse
        }
    
    def generate_blog_title(state: dict) -> dict:
        '''
        state: Contains current state of the graph (including 'blog_content').
        Returns a dict with the generated blog title.
        '''
        content = state.get("blog_content", "")

        system_msg = SystemMessage(content=f'''
        You are an expert LinkedIn content strategist and emotional copywriter.

        Your task is to analyze the following blog content and write a powerful, emotionally resonant title:

        "{content}"

        Guidelines:
        - 6–10 words
        - Not clickbaity
        - Must reflect tone and message of the blog
        ''')

        result = llm.invoke([system_msg])

        return {
            "messages": [result],
            "blog_title": result.content
        }
    
    def give_critic_agent(state: dict) -> dict:
        '''
        state: Contains current state of the graph (including 'blog_content' and 'blog_title').
        Returns a dict with the critic's feedback and routing decision.
        '''
        blog = state.get("blog_content", "")
        title = state.get("blog_title", "")

        system_msg = SystemMessage(content=f'''
        🎯 You are a thoughtful and emotionally intelligent content critic.

        You are reviewing a LinkedIn-style blog post and its suggested title.

        --- Blog Content ---
        {blog}

        --- Title ---
        {title}

        🧠 Please give detailed feedback on both:
        - Is the blog post well-written, structured, and valuable for LinkedIn readers?
        - Does the title capture the essence of the blog? Is it attention-worthy and emotionally resonant?

        🗺️ Based on your analysis, suggest where we should go next:
        - If the blog is weak or confusing, suggest regenerating the blog.
        - If only the title is weak or doesn't match the tone, suggest regenerating the title.
        - If both are good, confirm that no changes are needed.

        Your output should look like natural critique and end with a line saying:
        ➤ Route to: create_blog | create_title | END
        ''')

        result = llm.invoke([system_msg])
        
        return {
            "critic_feedback": result.content,
            "messages": [result]
        }
    
    tools = [generate_blog_agent, generate_blog_title, give_critic_agent]
    llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=False)

    sys_msg = SystemMessage(
        content=''' 

        You are the system entry point of the LangGraph agent flow.

        Your task is to receive a user-defined prompt and pass it to the next agent in the workflow by removing the unnessary words and by solving the grammar issues in the prompt if any.

        You do not generate content or perform analysis. Simply forward the prompt  to the next node in the chain for processing.

    '''
    )

    def assistant(state:State):
        return {"messages":[llm_with_tools.invoke([sys_msg]+state['messages'])]}
    
    def critic_condition(state: dict) -> str:
        feedback = state.get("critic_feedback", "")
        feedback_lower = feedback.lower()

        if "route to: create_blog" in feedback_lower:
            return "create_blog"
        elif "route to: create_title" in feedback_lower:
            return "create_title"
        elif "route to: end" in feedback_lower or "route to: the end" in feedback_lower:
            return "__end__"
        else:
            # Fallback to safe option
            return "__end__"

    builder2.add_node('create_blog',generate_blog_agent)
    builder2.add_node('create_title',generate_blog_title)
    builder2.add_node('give_critic',give_critic_agent)
    builder2.add_node('assistant', assistant)

    builder2.add_edge(START,'assistant')
    builder2.add_edge('assistant','create_blog')
    builder2.add_edge('create_blog','create_title')
    builder2.add_edge('create_title','give_critic')

    builder2.add_conditional_edges(
        "give_critic",
        critic_condition
    )

    agent = builder2.compile()
    return agent 


agent = create_blog_agent()
