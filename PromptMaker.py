# Core imports
import os
import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(filename='token_usage.log', level=logging.INFO)


# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found")

def read_prompt_template_from_file(file_path):
    """Load prompt template from file"""
    with open(file_path, 'r') as file:
        return file.read()
    
combined_prompt_template = read_prompt_template_from_file("Prompts/initialization_prompt")
    
# Init conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

# Create prompt template
method_prompt = PromptTemplate(
    input_variables=["query", "parent_method", "child_methods"],
    template=combined_prompt_template
)

document_prompt_template = read_prompt_template_from_file("Prompts/document_prompt")
document_prompt = PromptTemplate(
    input_variables=["Query", "Context"],
    template = document_prompt_template
)

# Init LLM
llm = ChatOpenAI(model="gpt-4")

# Setup sequence
method_sequence = method_prompt | llm
document_sequence = document_prompt | llm


def handle_user_query(selected_type, query, parent_method, child_methods, documents):
    if selected_type == "code":
        inputs = {
            "query": query,
            "parent_method": parent_method,
                "child_methods": child_methods,
            }
        sequence = method_sequence     
    elif selected_type == "document":
        inputs = {
            "Query": query,
            "Context": documents
        }
        sequence = document_sequence           
    with get_openai_callback() as callback:
        response = sequence.invoke(inputs)
        
        # Log tokens
        log_message = (
            f"Prompt Tokens: {callback.prompt_tokens}\n"
            f"Completion Tokens: {callback.completion_tokens}\n"
            f"Total Tokens: {callback.total_tokens}\n\n"
        )
        logging.info(log_message)               
         
    