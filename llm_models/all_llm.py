from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import OPENAI_API_KEY, DEEPSEEK_API_KEY, OPENAI_BASE_URL

llm = ChatOpenAI(
    temperature=0,
    model='qwen3.5-flash',
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

web_search_tool = TavilySearchResults(max_results=2)

# llm = ChatOpenAI(
#     temperature=0.5,
#     model='deepseek-chat',
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://api.deepseek.com")