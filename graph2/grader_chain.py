import json

from langchain_core.prompts import ChatPromptTemplate

from llm_models.all_llm import llm


# 提示词模板
system = """你是一个评估检索文档与用户问题相关性的评分器。
如果文档包含与用户问题相关的关键词或语义含义，则评为相关。
不需要非常严格的测试，目的是过滤掉错误的检索结果。
给出'yes'或'no'的二元评分来表示文档是否与问题相关。

请严格按照以下JSON格式返回结果，不要包含任何其他内容：
{{"binary_score": "yes"}} 或 {{"binary_score": "no"}}"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


def parse_grade_output(output) -> dict:
    """解析评分输出"""
    content = output.content if hasattr(output, 'content') else str(output)
    try:
        result = json.loads(content)
        if result.get("binary_score") not in ["yes", "no"]:
            result = {"binary_score": "yes"}
        return result
    except json.JSONDecodeError:
        return {"binary_score": "yes"}


# 构建检索评分器工作流
retrieval_grader_chain = grade_prompt | llm | parse_grade_output