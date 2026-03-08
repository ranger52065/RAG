# AGENTS.md

Guidance for AI coding agents working on this RAG enterprise knowledge base project.

## Project Overview

Python-based RAG system using LangChain, Milvus (hybrid search), OpenAI/DeepSeek LLMs, HuggingFace BGE-small-zh embeddings, LangGraph workflows, Loguru logging, and Tavily web search.

## Build/Run Commands

```bash
pip install -r requirements.txt

# Setup environment (copy .env.example to .env and configure)
# Required: OPENAI_API_KEY, DEEPSEEK_API_KEY, TAVILY_API_KEY
# Optional: MILVUS_URI (default: http://localhost:19530), OPENAI_BASE_URL

python main.py                       # Entry point (interactive mode)
python main.py --mode graph          # LangGraph workflow mode
python agent/rag_agent.py            # RAG agent standalone
python graph2/graph_2.py             # LangGraph workflow (interactive)
python documents/write_milvus.py     # Write docs to Milvus (multiprocessing)
```

## Testing

Tests are standalone scripts with `if __name__ == '__main__':` blocks. No pytest/unittest framework.

```bash
python search_tool/test_search.py                              # Run all tests in file
python -c "from search_tool.test_search import test9; test9()" # Run single test
python documents/markdown_parser.py --file path/to/file.md     # Parser with argparse
python documents/milvus_db.py                                  # Milvus connection test
```

## Linting/Type Checking

```bash
pip install ruff mypy
ruff check .
ruff format .
mypy . --ignore-missing-imports
```

## Project Structure

```
RAG_PROJECT/
├── agent/              # RAG agent with chat history
├── documents/          # Document parsing & Milvus operations
├── graph/              # LangGraph workflow v1
├── graph2/             # LangGraph workflow v2 (production)
├── llm_models/         # LLM & embedding models (module singletons)
├── tools/              # Agent tools (retriever)
├── search_tool/        # Search utilities & tests
├── utils/              # env_utils, log_utils, print_utils
├── datas/              # Data files (md/, output/)
└── main.py             # Entry point
```

## Code Style

### Imports

Grouped and separated by blank lines: stdlib → third-party → local imports

```python
import argparse
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus

from utils.env_utils import MILVUS_URI
from llm_models.embeddings_model import bge_embedding
```

### Naming Conventions

- `snake_case` - variables, functions, modules
- `PascalCase` - classes (`MilvusVectorSave`, `GraphState`)
- `UPPER_SNAKE_CASE` - constants (`OPENAI_API_KEY`, `COLLECTION_NAME`)
- `_prefix` - private methods
- lowercase - module singletons (`llm`, `log`, `bge_embedding`)

### Type Hints

```python
def parse_markdown_to_documents(self, md_file: str, encoding: str = 'utf-8') -> List[Document]:
    ...

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    ...
```

### Docstrings

Chinese docstrings for classes and key methods:

```python
class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中"""
```

### Error Handling

```python
try:
    docs = parser.parse_markdown_to_documents(file_path)
except Exception as e:
    log.error(f"解析失败 {file_path}: {str(e)}")
    log.exception(e)  # Includes full traceback
```

### Module Pattern

Each module can be run directly for testing:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Markdown文件解析器')
    parser.add_argument('--file', type=str, help='文件路径')
    args = parser.parse_args()
    # ... test code
```

## Key Patterns

### Environment Variables & Logging

```python
from dotenv import load_dotenv
load_dotenv(override=True)

from utils.log_utils import log

log.info(f"处理完成: {count} 个文档")
log.warning("警告：未找到任何.md文件")
log.error(f"写入数据失败！")
log.exception(e)  # Logs exception with traceback
```

### Module Singletons

Create at module level for reuse across the project:

```python
# llm_models/all_llm.py
llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=OPENAI_API_KEY)

# llm_models/embeddings_model.py
bge_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

### Milvus Hybrid Search

```python
from langchain_milvus import Milvus, BM25BuiltInFunction

vector_store = Milvus(
    embedding_function=bge_embedding,
    builtin_function=BM25BuiltInFunction(),
    vector_field=['dense', 'sparse'],
    connection_args={"uri": MILVUS_URI}
)

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 4, 'ranker_type': 'rrf', 'ranker_params': {'k': 100}, 'filter': {'category': 'content'}}
)
```

### Retriever Tool

```python
from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    '搜索并返回关于半导体的信息'
)
```

### RAG Chain (LCEL)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="根据上下文回答问题。\n问题：{question}\n上下文：{context}\n回答：",
    input_variables=["question", "context"],
)

rag_chain = prompt | llm | StrOutputParser()
generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
```

### LangChain Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableWithMessageHistory

agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[retriever_tool])

agent_with_history = RunnableWithMessageHistory(
    executor, get_session_history,
    input_messages_key='input', history_messages_key='chat_history'
)
```

### LangGraph Workflow

```python
from langgraph.graph import StateGraph
from langgraph.constants import START, END

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    transform_count: int

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_conditional_edges("generate", grade_generation, {"useful": END, "not useful": "transform_query"})
graph = workflow.compile()
```

### Multiprocessing

```python
from multiprocessing import Queue, Process

def producer(queue: Queue):
    queue.put(data)
    queue.put(None)  # Sentinel signal

def consumer(queue: Queue):
    while True:
        data = queue.get()
        if data is None:
            break
        process(data)

if __name__ == '__main__':
    q = Queue(maxsize=20)
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))
    p1.start(); p2.start()
    p1.join(); p2.join()
```

## Dependencies

Core: `langchain`, `langchain-community`, `langchain-core`, `langgraph`
Vector DB: `langchain-milvus`, `pymilvus`
LLM: `langchain-openai`, `langchain-huggingface`
Embeddings: `sentence-transformers`
Utilities: `loguru`, `python-dotenv`
Tools: `langchain-community` (TavilySearchResults)