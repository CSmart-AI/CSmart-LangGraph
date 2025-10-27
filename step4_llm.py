# Cell 14
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from dotenv import load_dotenv
import os

# .env 파일에서 GOOGLE_API_KEY 불러오기
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# 기본 LLM - Gemini 사용
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0,
    streaming=True
)

# 도구 바인딩
from step3_db_and_search import tools
llm_with_tools = llm.bind_tools(tools)
