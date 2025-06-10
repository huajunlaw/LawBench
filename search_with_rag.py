import argparse
import sys

from langchain_community.llms import VLLMOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from loguru import logger

from import_china_laws_to_vector_store import index_documents


def get_embedding_function(model="Qwen3-Embedding-8B"):
    """."""
    embeddings = OpenAIEmbeddings(model=model, base_url='http://localhost:8001/v1',)
    # With the `text-embedding-3` class of models, you can specify the size of the embeddings you want returned.
    # dimensions=1024
    return embeddings


def create_rag_chain(vector_store, llm_model_name="Qwen3-8B-Base", context_window=8192):
    """Creates the RAG chain."""
    # Initialize the LLM

    llm = VLLMOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1",
                     model_name=llm_model_name,
                     # model_kwargs={"stop": ["."]},
                     )

    logger.info(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Or "mmr"
        search_kwargs={'k': 3}  # Retrieve top 3 relevant chunks
    )
    logger.info("Retriever initialized.")

    # Define the prompt template
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    logger.info("RAG chain created.")
    return rag_chain


def query_rag(chain, question):
    """Queries the RAG chain and logger.infos the response."""
    logger.info("\nQuerying RAG chain...")
    logger.info(f"Question: {question}")
    response = chain.invoke(question)
    logger.info("\nResponse:")
    logger.info(response)


def get_vector_store(embedding_function, connection="", collection_name=""):
    """Initializes or loads the  vector store."""
    vector_store = PGVector(
    embeddings=embedding_function,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
    return vector_store


def main(argv):
    """."""
    # 0. constant
    connection = "postgresql+psycopg://batchcom:@localhost:5432/postgres"
    collection_name = "all_laws_china"

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    commit_parser = subparsers.add_parser("index", help="导入向量数据库")
    commit_parser.add_argument("-p", "--path", help="原始文件所在路径", required=True)

    parser.add_argument("-m", "--model", dest="model", help="LLM model: it should be a str of serverd_name.")
    parser.add_argument("-e", "--embed", dest="embed", help="Embedding model: it should be a str of serverd_name")
    args = parser.parse_args(argv)
    logger.info(args)
    # 3. Get Embedding Function
    embed = args.embed or "Qwen3-Embedding-8B"
    embedding_function = get_embedding_function(embed)

    # 4. Index Documents (Only needs to be done once per document set)
    # Check if DB exists, if not, index. For simplicity, we might re-index here.
    # A more robust approach would check if indexing is needed.
    if args.command == 'index':
        DATA_PATH = args.path
        index_documents(DATA_PATH, embedding_function, connection=connection, collection_name=collection_name)
    # To load existing DB instead:
    vector_store = get_vector_store(embedding_function, connection=connection, collection_name=collection_name)
    # 5. Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name=args.model)  # Use the chosen Qwen 3 model

    # 6. Query
    query_question = "What is the main topic of the document?"  # Replace with a specific question
    query_rag(rag_chain, query_question)

    query_question_2 = "Summarize the introduction section."  # Another example
    query_rag(rag_chain, query_question_2)


if __name__ == "__main__":
    main(sys.argv[1:])
