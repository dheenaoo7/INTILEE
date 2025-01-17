
import os 
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = ("Enter API key for OpenAI: ")
else:
  print("API key is already set": os.environ.get("OPENAI_API_KEY"))