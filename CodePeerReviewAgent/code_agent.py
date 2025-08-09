from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel
import os

# Load variables from .env file into environment
load_dotenv()
import os 
API_KEYS = ['GROQ_API_KEY','GROQ_YOUTUBE_COLLAB','HUGGINGFACE_TOKEN','LANGCHAIN_API_KEY','LANGCHAIN_PROJECT','HF_TOKEN','GOOGLE_API_KEY']
for k in API_KEYS:
    value = os.getenv(k)
    if value:
        os.environ[k] = value
        print(f"{k} loaded successfully.")
    else:
        print(f"{k} not found in .env or environment.")
        
from langchain_groq import ChatGroq
llm = ChatGroq(
    model='Llama3-8b-8192',
)

    
def code_agent():
    from typing import Optional
    from pydantic import BaseModel

    class ChatState(BaseModel):
        user_input: str  # This is the only required field at the start
        code: Optional[str] = None
        test_cases: Optional[str] = None
        output_file: Optional[str] = None
        review: Optional[str] = None
        documentation: Optional[str] = None
        manager: Optional[str] = None
        router_decision: Optional[str] = None # Add this new field for the router's output
    def generate_code_agent(state: ChatState) -> dict:
        """
        Generates production-ready code for the problem described in `state.user_input`.
        If no language is defined use Python by default.
        Return the code.
        """
        try:
            prompt = f"""
            You are a code-generation agent.
            Your task is to write **production-grade, well-documented code** to solve the following problem:
            
            Problem Statement:
            {state.user_input}
            
            Requirements:
            - Use the programming language mentioned in the problem, if specified. 
            If no language is mentioned, default to Python.
            - Ensure the code is clean, readable, and maintainable.
            - Add comments explaining key logic steps.
            - Include input validation and handle edge cases gracefully.
            - Avoid unnecessary complexity — keep it efficient.
            
            Output only the complete code, no explanations.
            """
            
            task = llm.invoke(prompt)

            # Ensure the output is a string and Pydantic-compatible
            if not hasattr(task, "content") or not isinstance(task.content, str):
                raise ValueError("Invalid LLM output format: 'content' missing or not a string.")

            return {
                "code": task.content.strip()
            }

        except Exception as e:
            # Log or handle error, but keep it Pydantic-friendly
            return {
                "code": f"# ERROR: Failed to generate code.\n# Reason: {str(e)}"
            }
        
    def generate_test_cases_agent(state: ChatState) -> dict:
        """
        You have to generate a comprehensive set of test cases for the user_input problem in the same language defined by user in prompt. If no language is defined use Python by default.
        Returns the test case.
        """
        try:
            prompt = f"""
            You are a test-case generation agent.
            Your task is to produce a **comprehensive set of test cases** for the solution to the following problem:
            
            Problem Statement:
            {state.user_input}
            
            Requirements:
            - Use the **same programming language** as the code from `generate_code_agent`.
            - Cover normal, boundary, and edge cases.
            - Include both positive and negative test scenarios.
            - Ensure tests are self-contained and runnable.
            - Where applicable, include automated unit tests using the language's standard testing framework.
            
            Output only the test code, no explanations.
            """

            task = llm.invoke(prompt)

            # Ensure LLM output format is correct
            if not hasattr(task, "content") or not isinstance(task.content, str):
                raise ValueError("Invalid LLM output format: 'content' missing or not a string.")

            return {
                "test_cases": task.content.strip()
            }

        except Exception as e:
            # Always return a valid field value so Pydantic validation passes
            return {
                "test_cases": f"# ERROR: Failed to generate test cases.\n# Reason: {str(e)}"
            }


    def generate_documentation_agent(state: ChatState) -> dict:
        """
        Generates clear and structured documentation for the code solving `state.user_input`.
        Includes robust error handling to ensure Pydantic-friendly return values.
        Returns the documentation.
        """
        try:
            prompt = f"""
            You are a documentation-generation agent .
            Your task is to write **clear, user-friendly documentation** for the solution to the following problem:
            
            Problem Statement:
            {state.user_input}
            
            Requirements:
            - Use Markdown format.
            - Include a short problem description.
            - Document function/class parameters, return values, and exceptions.
            - Add at least one example usage.
            - Mention any important constraints or assumptions.
            - Keep it concise but complete.
            
            Output only the documentation, no explanations.
            """

            task = llm.invoke(prompt)

            # Ensure output format is valid
            if not hasattr(task, "content") or not isinstance(task.content, str):
                raise ValueError("Invalid LLM output format: 'content' missing or not a string.")

            return {
                "documentation": task.content.strip()
            }

        except Exception as e:
            # Always return a valid key for Pydantic compatibility
            return {
                "documentation": f"# ERROR: Failed to generate documentation.\n# Reason: {str(e)}"
            }


    def combine_all_file(state: ChatState) -> dict:
        """
        Combines all generated outputs into a single output file.
        Returns only the 'output_file'.
        """
        try:
            # Build combined content
            combine = []
            combine.append(f"Here is the complete file about {state.user_input}!\n")
            combine.append(f"# Problem Statement:\n{state.user_input}\n")
            combine.append(f"# Generated Code:\n{state.code}\n")
            combine.append(f"# Test Cases:\n{state.test_cases}\n")
            combine.append(f"# Documentation:\n{state.documentation}\n")

            return {
                "output_file": "\n".join(combine).strip()
            }

        except Exception as e:
            # Fallback to error message while keeping it Pydantic-valid
            return {
                "output_file": f"# ERROR: Failed to combine output.\n# Reason: {str(e)}"
            }


    def evaluator_agent(state: ChatState) -> dict:
        """
        Evaluates the generated code, test cases, and documentation for the problem in `state.user_input`.
        Returns only the 'review'.
        """
        try:
            prompt = f"""
            You are a code evaluator agent in an orchestrator system.
            Your job is to review and evaluate the following solution to the given problem.

            Problem Statement:
            {state.user_input}

            Generated Code:
            {state.code}

            Generated Test Cases:
            {state.test_cases}

            Generated Documentation:
            {state.documentation}

            Evaluation Requirements:
            - Assess **correctness**: Does the code solve the problem as stated?
            - Assess **completeness**: Are all required features and constraints addressed?
            - Assess **code quality**: Maintainability, readability, and adherence to best practices.
            - Assess **test coverage**: Do the test cases check normal, boundary, and error conditions?
            - Assess **documentation quality**: Clarity, accuracy, and helpfulness.
            - Identify **edge cases** not covered in the code or tests.
            - Provide **specific improvement suggestions**.
            - Rate each category on a scale of 1–10.

            Output a structured review in the following format:
            ---
            **Evaluation Report**
            Correctness: X/10 - <comments>
            Completeness: X/10 - <comments>
            Code Quality: X/10 - <comments>
            Test Coverage: X/10 - <comments>
            Documentation: X/10 - <comments>
            Suggested Improvements:
            - Point 1
            - Point 2
            - ...
            ---
            """

            task = llm.invoke(prompt)

            # Validate LLM output
            if not hasattr(task, "content") or not isinstance(task.content, str):
                raise ValueError("Invalid LLM output format: 'content' missing or not a string.")

            return {
                "review": task.content.strip()
            }

        except Exception as e:
            # Return a Pydantic-compatible error string
            return {
                "review": f"# ERROR: Failed to evaluate solution.\n# Reason: {str(e)}"
            }


    def manager_agent(state: ChatState) -> dict:
        """
        Provides contextual feedback on the evaluation report for the router agent.

        This agent:
        - Receives the evaluation report from `evaluator_agent`.
        - Summarizes or clarifies the key points in a structured, concise way.
        - Ensures the feedback is clean and unambiguous for `llm_call_router` to use
            when making the routing decision.

        It does NOT decide the next pipeline step — that is the responsibility of
        `llm_call_router`.

        Args:
            state (ChatState): Shared conversation state containing:
                            - `user_input`: Original problem statement.
                            - `review`: Evaluation report from `evaluator_agent`.

        Returns:
        Contains the key feedback points in a structured format.
        The feedback will be passed to `llm_call_router` for decision-making.
        """
        prompt = f"""
        You are the manager .
        You have received this evaluation report from the evaluator agent:

        Evaluation Report:
        {state.review}

        Your task:
        - Summarize the key findings in 2–5 concise bullet points.
        - Include any clarifications needed for the routing decision agent (`llm_call_router`).
        - Do NOT decide the action or next step yourself.
        - Avoid vague language — the feedback must be clear, objective, and decision-ready.

        Output ONLY in this format:
        ---
        **Manager Feedback**
        - Point 1
        - Point 2
        - ...
        ---
        """

        task = llm.invoke(prompt)
        return {"manager": task.content.strip()}


    from typing_extensions import Literal
    from pydantic import BaseModel, Field
    from langchain_core.messages import HumanMessage, SystemMessage

    class RouteDecision(BaseModel):
        step: Literal[
            "generate_code_agent",
            "generate_test_cases_agent",
            "generate_documentation_agent",
        
        ] = Field(
            ...,
            description="The next step in the orchestrator pipeline"
        )


    router = llm.with_structured_output(RouteDecision)

    def llm_call_router(state: ChatState) -> dict:
        # Use a simpler, more direct prompt for the LLM
        decision_prompt = f"""
        Based on the following manager feedback and original evaluation report,
        decide the next step in the pipeline.
        Manager Feedback: {state.manager}
        Original Evaluation Report: {state.review}
        """

        task = router.invoke([
            SystemMessage(content="You are a routing agent. Your job is to select the next action based on the feedback provided."),
            HumanMessage(content=decision_prompt)
        ])
        
        # task is now a RouteDecision Pydantic object
        return {"router_decision": task.step}
    
    def route_decision(state: ChatState) -> str:
        # The llm_call_router already provides the exact node name
        step = state.router_decision
        return step
    
    parallel_builder = StateGraph(ChatState)


    parallel_builder.add_node('generate_code_agent', generate_code_agent)
    parallel_builder.add_node('generate_test_cases_agent', generate_test_cases_agent)
    parallel_builder.add_node('generate_documentation_agent', generate_documentation_agent)
    parallel_builder.add_node('output_file', combine_all_file)
    parallel_builder.add_node('evaluator_agent', evaluator_agent)
    parallel_builder.add_node('manager_agent', manager_agent)
    parallel_builder.add_node('llm_call_router', llm_call_router)

    parallel_builder.add_edge(START, 'generate_code_agent')
    parallel_builder.add_edge(START, 'generate_test_cases_agent')
    parallel_builder.add_edge(START, 'generate_documentation_agent')

    parallel_builder.add_edge('generate_code_agent', 'output_file')
    parallel_builder.add_edge('generate_test_cases_agent', 'output_file')
    parallel_builder.add_edge('generate_documentation_agent', 'output_file')
    parallel_builder.add_edge('output_file','evaluator_agent')
    parallel_builder.add_edge('evaluator_agent', 'manager_agent')
    parallel_builder.add_edge('manager_agent', 'llm_call_router')
    parallel_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "generate_code_agent": "generate_code_agent",
            "generate_test_cases_agent": "generate_test_cases_agent",
            "generate_documentation_agent": "generate_documentation_agent",
            "evaluator_agent": "evaluator_agent",
            "APPROVED": END # If the router says 'APPROVED', go to END
        }
    )
    parallel_builder.add_edge('manager_agent', END)

    from IPython.display import Image,display
    from langgraph.checkpoint.memory import MemorySaver

    # Compile workflow
    coder_agent = parallel_builder.compile()

    # Show the workflow
    
    return coder_agent


agent = code_agent()