import json

from langchain_core.prompts import ChatPromptTemplate

from llm_models.all_llm import llm


# 查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）

# 提示词模板
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档。
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。

请严格按照以下JSON格式返回结果，不要包含任何其他内容：
{{"datasource": "vectorstore"}} 或 {{"datasource": "web_search"}}"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


def parse_route_output(output) -> dict:
    """解析路由输出"""
    content = output.content if hasattr(output, 'content') else str(output)
    try:
        result = json.loads(content)
        if result.get("datasource") not in ["vectorstore", "web_search"]:
            result = {"datasource": "vectorstore"}
        return result
    except json.JSONDecodeError:
        return {"datasource": "vectorstore"}


# 创建问题路由器链
question_router_chain = route_prompt | llm | parse_route_output