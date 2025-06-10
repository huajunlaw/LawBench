import argparse
import os
import sys
import json
from functools import partial


from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

os.environ["OPENAI_API_KEY"] = "EMPTY"


def get_embedding_function(model="Qwen3-Embedding-8B"):
    """."""
    embeddings = OpenAIEmbeddings(model=model, base_url='http://localhost:8001/v1',)
    # With the `text-embedding-3` class of models, you can specify the size of the embeddings you want returned.
    # dimensions=1024
    return embeddings


def metadata_func(record: dict, metadata: dict, meta_type) -> dict:
    metadata["type"] = meta_type
    metadata["content"] = json.dumps(record, ensure_ascii=False)
    return metadata


def load_documents(file_path):
    """Loads documents from the specified data path."""
    content_key_level = '. | [.level1, .level2,.level3] | join("\n")| walk(if type == "string" then gsub("#"; "") else . end)'
    loader_level = JSONLoader(file_path=file_path, jq_schema='.[]', content_key=content_key_level, is_content_key_jq_parsable=True, metadata_func=partial(metadata_func, meta_type='levels'), text_content=False)
    loader_desc = JSONLoader(file_path=file_path, jq_schema='.[]', content_key='desc', text_content=False, metadata_func=partial(metadata_func, meta_type='desc'))
    documents = loader_level.load() + loader_desc.load()
    logger.info(f"Loaded {len(documents)} page(s) from {file_path}")
    return documents


def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(all_splits)} chunks")
    return all_splits


def index_documents(data_path, embedding_function, connection="", collection_name=""):
    """Indexes document chunks into the Chroma vector store."""
    # 1. Load Documents
    for x in os.listdir(data_path):
        file_path = os.path.join(data_path, x)
        docs = load_documents(file_path)
        # 2. Split Documents
        chunks = split_documents(docs)
        logger.info("Attempting to index documents...")
        logger.info(f"Indexing {len(chunks)} chunks...")
        PGVector.from_documents(
            documents=chunks,
            embedding=embedding_function,
            connection=connection,
            collection_name=collection_name
        )


def main(argv):
    """."""
    # 1. constant
    connection = "postgresql+psycopg://batchcom:@localhost:5432/postgres"
    collection_name = "all_laws_china"

    # 2. args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="原始文件所在路径", required=True)
    parser.add_argument("-e", "--embed", dest="embed", help="embed model: it should be a str ")
    args = parser.parse_args(argv)
    logger.info(args)

    # 3. Get Embedding Function
    embed = args.embed or "Qwen3-Embedding-8B"
    embedding_function = get_embedding_function(embed)

    # 4. Index Documents (Only needs to be done once per document set)
    DATA_PATH = args.path
    index_documents(DATA_PATH, embedding_function, connection=connection, collection_name=collection_name)


if __name__ == "__main__":
    main(sys.argv[1:])
