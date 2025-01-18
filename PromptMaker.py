# Updated imports
import os
import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # Please check for the latest memory class

# Check for the OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found. Please set it in your environment variables.")

# Initialize the ChatOpenAI model from the langchain-openai package
llm = ChatOpenAI(model="gpt-4")

# Set up logging to capture token usage
logging.basicConfig(filename='token_usage.log', level=logging.INFO)

# Define the combined context prompt template
combined_prompt_template = """Generate a concise response for a query based on the provided parent method and child methods.
Focus only on the response content, and keep it as concise as possible.

Query: {query}
Parent Method: {parent_method}
Child Methods: {child_methods}"""

# Set up memory to keep track of the conversation (check for new memory class)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

# Create a prompt template that uses the combined context string
prompt = PromptTemplate(
    input_variables=["query", "parent_method", "child_methods"],  # Make sure the variables match
    template=combined_prompt_template
)

# Create a RunnableSequence instead of LLMChain
sequence = prompt | llm

# Function to get the response with logging and token tracking
def get_code_assistant_response(query, parent_method, child_methods):
    # Prepare the inputs with the necessary context
    inputs = {
        "query": query,
        "parent_method": parent_method,
        "child_methods": child_methods,
    }

    # Callback to capture token usage during the chain invocation
    with get_openai_callback() as callback:
        response = sequence.invoke(inputs)  # Using the new RunnableSequence approach
        
        # Log the token usage to the file
        log_message = (
            f"Prompt Tokens: {callback.prompt_tokens}\n"
            f"Completion Tokens: {callback.completion_tokens}\n"
            f"Total Tokens: {callback.total_tokens}\n\n"
        )
        logging.info(log_message)

    return response
