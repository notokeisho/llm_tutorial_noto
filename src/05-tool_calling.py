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

# 出力の構造を定義
"""
-名前
-生きていた時代
    1980~2000
-どんな人か
   概要
-実績

"""


class Achievement(BaseModel):
    year: str = Field(description="年(ex. 1790)")
    content: str = Field(description="内容")


class Profile(BaseModel):
    name: str = Field(description="人物の名前")
    lifespan: str = Field(description="人物の生存期間(ex. 1780~1891)")
    overview: str = Field(description="人物の概要")
    achievements: List[Achievement] = Field(description="人物の実績")


def extract_person_info(llm: ChatOpenAI, input: str) -> Profile:

    parser = PydanticOutputParser(pydantic_object=Profile)

    prompt = PromptTemplate(
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
        次の文章から人物の情報を抽出してください。
        #人物の情報
        {input}
        
        #出力形式
        {format_instructions}
       """,
    )

    # prompt => llm => parserの順でやっていく
    chain = prompt | llm | parser

    profile: Profile = chain.invoke({"input": input})

    return profile


def reserch_person_info(llm: ChatOpenAI, person: str) -> str:
    # ツール作成
    tools = [TavilySearchResults(max_result=5)]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "人物についての情報を調査してください。生存期間や業績などの詳細を調査してください。必要に応じて、Tavily_search_result_jsonをを呼びます",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor.invoke({"input": person})["output"]

    pass


def main():
    person = input("歴史上の人物名を入力してください: ")
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    report = reserch_person_info(llm, person)
    profile = extract_person_info(llm, report)

    # print(profile.model_dump_json)
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
