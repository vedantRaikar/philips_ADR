import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from context_med import get_wikipedia_documents, get_arxiv_documents

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_TOKEN")
M_API_KEY = os.getenv("MISTRAL_API_KEY")

def create_vectorstore_from_text(input_text, persist_directory=None, chunk_size=500, chunk_overlap=50):
    """
    Creates a Chroma vector store from text input with persistence option.
    
    Args:
        input_text (str): The text to vectorize
        persist_directory (str, optional): Directory to persist the vectorstore
        chunk_size (int): Size of text chunks (default: 500)
        chunk_overlap (int): Overlap between chunks (default: 50)
    """
    try:
        # Step 1: Convert raw text to a LangChain Document
        raw_documents = [Document(page_content=input_text)]

        # Step 2: Split the text into chunks with better settings
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"
        )
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"Created {len(documents)} chunks from input text")

        # Initialize Mistral AI embeddings
        embed_model = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=M_API_KEY,
            max_retries=3
        )
        
        vectorstore = Chroma.from_documents(documents, embed_model)
        print("Created in-memory vector store")
        return vectorstore

    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        raise


def query_vectorstore(query, vectorstore):
    """
    Queries the in-memory Chroma vector store.
    """
    embed_model = MistralAIEmbeddings(model="mistral-embed", api_key=M_API_KEY)
    query_vector = embed_model.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_vector, k=3)
    return results


# ✅ Example usage
# if __name__ == "__main__":
#     input_text = "\n\n".join([doc.page_content for doc in get_wikipedia_documents("aspirin") + get_arxiv_documents("aspirin")])
#     query = "what is aspirin used for?"
#     vectorstore = create_vectorstore_from_text(input_text)
#     docs = query_vectorstore(query, vectorstore)
#     for doc in docs:
#         print(doc.page_content)
