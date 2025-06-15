from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import json
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults

# 環境変数の読み込み
load_dotenv()

# 出力の構造と定義
"""
-アーティスト名
-代表曲
-曲一覧
"""


class Discography(BaseModel):
    year: str = Field(description="年(ex. 1790)")
    song: str = Field(description="曲名")


# クラスを定義
class Profile(BaseModel):
    artist: str = Field(description="アーティスト名")
    hitSong: Discography = Field(description="ヒット曲")
    Discographys: List[Discography] = Field(description="リリース履歴")
    
def main():
    #LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    
    #Output parser
    parser = PydanticOutputParser(pydantic_object=Profile)
    
    #prompt template
    prompt = PromptTemplate(
        partial_variables={"format_instructions":par}
        template="""
        ミュージシャン、アーティスト{artist}について楽曲と代表曲を教えてください。
        {format_instructions}
        回答は必ず指定された形式で出力してください。
        """
    )