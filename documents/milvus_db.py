import argparse
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, DataType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self):
        self.vector_store_saved: Optional[Milvus] = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        
        if COLLECTION_NAME in client.list_collections():
            client.drop_collection(collection_name=COLLECTION_NAME)

        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_name="dense_idx",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 64}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            vector_field='dense',
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_documents(self, datas: List[Document]):
        self.vector_store_saved.add_documents(datas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milvus向量数据库操作')
    parser.add_argument('--file', type=str, help='要解析的Markdown文件路径')
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'md', 'sample.md')

    parser_md = MarkdownParser()
    docs = parser_md.parse_markdown_to_documents(file_path)

    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    desc_collection = client.describe_collection(collection_name=COLLECTION_NAME)
    print('表结构是: ', desc_collection)

    res = client.list_indexes(collection_name=COLLECTION_NAME)
    print('表中的所有索引：', res)

    if res:
        for i in res:
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",
        output_fields=['text', 'category', 'filename']
    )

    print('测试 过滤查询的结果是: ', result)