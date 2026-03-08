import json

from langchain_core.prompts import ChatPromptTemplate

from llm_models.all_llm import llm


# 提示词模板
system = """您是一个评估回答是否解决用户问题的评分器。
给出'yes'或'no'的二元评分。'yes'表示:回答确实解决了该问题。

请严格按照以下JSON格式返回结果，不要包含任何其他内容：
{{"binary_score": "yes"}} 或 {{"binary_score": "no"}}"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题: \n\n {question} \n\n 生成回答: {generation}"),
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


# 构建回答质量评估工作流
answer_grader_chain = answer_prompt | llm | parse_grade_output