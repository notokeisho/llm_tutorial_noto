from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import json

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


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # Output parser
    parser = PydanticOutputParser(pydantic_object=Profile)

    # prompt template
    prompt = PromptTemplate(
        input_variables=["person"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
        歴史学の学習で歴史上の人物について研究しています。
        歴史上の人物{person}についてのプロフィールを作成してください。
        {format_instructions}
        回答は必ず指定された形式で出力してください。
        """,
    )

    # prompt => llm => parserの順でやっていく
    chain = prompt | llm | parser

    person = input("歴史上の人物名を入力してください: ")

    profile: Profile = chain.invoke({"person": person})

    # print(profile.model_dump_json)
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
