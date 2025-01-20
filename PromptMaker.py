import os
import logging
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(filename='token_usage.log', level=logging.INFO)

# Check API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables. Please set it before running the script.")

def read_prompt_template_from_file(file_path):
    """Load prompt template from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")
    with open(file_path, 'r') as file:
        return file.read()

# Load prompt templates
combined_prompt_template = read_prompt_template_from_file("Prompts/initialization_prompt")
document_prompt_template = read_prompt_template_from_file("Prompts/document_prompt")

# Init conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

# Create prompt templates
method_prompt = PromptTemplate(
    input_variables=["query", "parent_method", "child_methods"],
    template=combined_prompt_template
)

document_prompt = PromptTemplate(
    input_variables=["Query", "Context"],
    template=document_prompt_template
)

# Init LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

def handle_user_query(selected_type, query, parent_method=None, child_methods=None, documents=None):
    """
    Handles the user's query and determines the appropriate chain to use.
    """
    # Validate input
    if not query:
        raise ValueError("You must provide a query.")
    if selected_type == "KapCode" and not (parent_method and child_methods):
        raise ValueError("For KapCode, 'parent_method' and 'child_methods' are required.")
    if selected_type == "KapDoc" and not documents:
        raise ValueError("For KapDoc, 'documents' cannot be empty.")

    # Prepare inputs based on type
    if selected_type == "KapCode":
        inputs = {
            "query": query,
            "parent_method": parent_method,
            "child_methods": ", ".join(child_methods) if child_methods else ""
        }
        # Generate prompt using the method_prompt template
        prompt = method_prompt.format(**inputs)
    elif selected_type == "KapDoc":
        inputs = {
            "query": query,
            "context": "\n".join(documents) if documents else ""
        }
        # Generate prompt using the document_prompt template
        prompt = document_prompt.format(**inputs)
    else:
        raise ValueError(f"Invalid query type: {selected_type}. Must be 'KapCode' or 'KapDoc'.")

    # Invoke the LLM model with the generated prompt and log token usage
    with get_openai_callback() as callback:
        response = llm.invoke(prompt)  # Directly invoke the LLM model with the prompt
        
        log_message = (
            f"Prompt Tokens: {callback.prompt_tokens}\n"
            f"Completion Tokens: {callback.completion_tokens}\n"
            f"Total Tokens: {callback.total_tokens}\n\n"
        )
        logging.info(log_message)
    
    return response.content
