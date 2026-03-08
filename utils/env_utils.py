import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 't_collection01')
MD_DATA_DIR = os.getenv('MD_DATA_DIR', './datas/md')
