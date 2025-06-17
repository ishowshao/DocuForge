import dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

dotenv.load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Pydantic
class StringArray(BaseModel):
    """Array of strings."""
    
    items: list[str] = Field(description="List of string items")

structured_llm = llm.with_structured_output(StringArray)

messages = [
    SystemMessage("请把用户输入的句子合理的断开成2段，每段不超过10个字。不可以修改原文。最终以 JSON 字符串数组形式返回。"),
    HumanMessage("人工智能，指由人制造出来的机器所表现出来的智能。"),
]

response = structured_llm.invoke(messages)

print(response.items)


