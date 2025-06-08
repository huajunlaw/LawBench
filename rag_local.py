import os

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader # Or UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import VLLMOpenAI


from loguru import logger


def get_embedding_function(model_path="Qwen3-Embedding-0.6B-Q8_0.gguf"):
    """."""
    return LlamaCppEmbeddings(model_path=model_path)  # type:ignore


def load_documents(DATA_PATH, PDF_FILENAME):
    """Loads documents from the specified data path."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    # loader = UnstructuredPDFLoader(pdf_path) # Alternative
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} page(s) from {pdf_path}")
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


def create_rag_chain(vector_store, llm_model_name="Qwen3-8B-Base", context_window=8192):
    """Creates the RAG chain."""
    # Initialize the LLM

    llm = VLLMOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1",
                     model_name=llm_model_name,
                     model_kwargs={"stop": ["."]},
                     )

    print(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or "mmr"
        search_kwargs={'k': 3} # Retrieve top 3 relevant chunks
    )
    print("Retriever initialized.")

    # Define the prompt template
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain


def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    print("\nQuerying RAG chain...")
    print(f"Question: {question}")
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)


def get_vector_store(embedding_function, connection="", collection_name=""):
    """Initializes or loads the  vector store."""
    vector_store = PGVector(
    embeddings=embedding_function,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
    return vector_store


def index_documents(chunks, embedding_function, vector_store: PGVector):
    """Indexes document chunks into the Chroma vector store."""
    logger.info(f"Indexing {len(chunks)} chunks...")
    vectorstore = vector_store.from_documents(
        documents=chunks,
        embedding=embedding_function,
    )
    return vectorstore


def main():
    """."""
    # 1. Load Documents
    DATA_PATH = "data/"
    PDF_FILENAME = "llama2.pdf" # Replace with your PDF filename
    connection = "postgresql+psycopg://batchcom:@localhost:5432/postgres"
    collection_name = "all_laws_china"
    docs = load_documents(DATA_PATH, PDF_FILENAME)

    # 2. Split Documents
    chunks = split_documents(docs)

    # 3. Get Embedding Function
    embedding_function = get_embedding_function() # Using Ollama nomic-embed-text

    # 4. Index Documents (Only needs to be done once per document set)
    # Check if DB exists, if not, index. For simplicity, we might re-index here.
    # A more robust approach would check if indexing is needed.
    print("Attempting to index documents...")
    # To load existing DB instead:
    vector_store = get_vector_store(embedding_function, connection=connection, collection_name=collection_name)
    vector_store = index_documents(chunks, embedding_function, vector_store)

    # 5. Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b") # Use the chosen Qwen 3 model

    # 6. Query
    query_question = "What is the main topic of the document?" # Replace with a specific question
    query_rag(rag_chain, query_question)

    query_question_2 = "Summarize the introduction section." # Another example
    query_rag(rag_chain, query_question_2)

if __name__ == "__main__":
    main()
