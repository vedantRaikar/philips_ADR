import requests
import xml.etree.ElementTree as ET
import tiktoken
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from langchain.schema import Document

# Maximum token limit for context
MAX_TOKENS = 2000

# Initialize LangChain retrievers
wikipedia_retriever = WikipediaRetriever()
arxiv_retriever = ArxivRetriever()

def get_context(drug_name, pubmed_limit=5):
    """
    Retrieve comprehensive information about a drug, including its PubChem data,
    Wikipedia documents, arXiv articles, and related PubMed articles, ensuring the
    total token count does not exceed the model's context window.

    Parameters:
    - drug_name (str): The common name of the drug.
    - pubmed_limit (int): Maximum number of PubMed articles to retrieve.

    Returns:
    - dict: A dictionary containing the drug's context information.
    """
    context = {"Drug Name": drug_name}

    context["PubChem Data"] = get_pubchem_data(drug_name)
    context["Wikipedia Articles"] = get_wikipedia_documents(drug_name)
    context["arXiv Articles"] = get_arxiv_documents(drug_name)
    context["PubMed Articles"] = get_pubmed_articles(drug_name, pubmed_limit)

    truncated_context = truncate_context(context, MAX_TOKENS)

    return truncated_context

def get_pubchem_data(drug_name):
    """Fetch chemical properties from PubChem."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        properties = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        return {
            "Molecular Formula": properties.get("MolecularFormula", "N/A"),
            "Molecular Weight": properties.get("MolecularWeight", "N/A"),
            "IUPAC Name": properties.get("IUPACName", "N/A"),
            "Canonical SMILES": properties.get("CanonicalSMILES", "N/A")
        }
    return {"Error": "Unable to fetch PubChem data."}



def get_wikipedia_documents(drug_name: str, doc_content_chars_max: int = 1000) -> list[Document]:
    """
    Retrieve Wikipedia documents about a drug.
    
    Args:
        drug_name: Name of the drug to search for
        doc_content_chars_max: Maximum character count for document content
        
    Returns:
        List of Document objects containing Wikipedia content
    """
    try:
        wikipedia_retriever = WikipediaRetriever(
            lang="en",
            doc_content_chars_max=doc_content_chars_max,
            top_k_results=3,
        )
        # Use the newer invoke() method instead of get_relevant_documents()
        docs = wikipedia_retriever.invoke(drug_name)
        
        if not docs:
            print(f"No Wikipedia documents found for: {drug_name}")
            return []
            
        return docs
    except Exception as e:
        print(f"Error retrieving Wikipedia documents: {e}")
        return []

def get_arxiv_documents(drug_name: str, max_docs: int = 3) -> list[Document]:
    """
    Retrieve ArXiv documents about a drug.
    
    Args:
        drug_name: Name of the drug to search for
        max_docs: Maximum number of documents to retrieve
        
    Returns:
        List of Document objects containing ArXiv content
    """
    try:
        arxiv_retriever = ArxivRetriever(
            load_max_docs=max_docs,
            load_all_available_meta=True,
            doc_content_chars_max=1000,
        )
        # Use the newer invoke() method instead of get_relevant_documents()
        query = f"medical {drug_name}"
        docs = arxiv_retriever.invoke(query)
        
        if not docs:
            print(f"No ArXiv documents found for: {drug_name}")
            return []
            
        return docs
    except Exception as e:
        print(f"Error retrieving ArXiv documents: {e}")
        return []

def get_pubmed_articles(drug_name, limit=2):
    """Fetch related articles from PubMed."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": drug_name,
        "retmax": limit,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    articles = []
    if response.ok:
        root = ET.fromstring(response.content)
        ids = [id_elem.text for id_elem in root.findall(".//Id")]
        for pubmed_id in ids:
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": pubmed_id,
                "retmode": "json"
            }
            fetch_response = requests.get(fetch_url, params=fetch_params)
            if fetch_response.ok:
                summary = fetch_response.json().get("result", {}).get(pubmed_id, {})
                articles.append({
                    "Title": summary.get("title", "No title available"),
                    "Source": summary.get("source", "No source available"),
                    "Publication Date": summary.get("pubdate", "No date available"),
                    "PubMed ID": pubmed_id,
                    "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                })
    return articles

def truncate_context(context, max_tokens=3000):
    """Truncate the context to fit within the max token limit."""
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(tokenizer.encode(text))

    sections = {}
    total_tokens = 0
    for key, value in context.items():
        if isinstance(value, list):
            serialized_value = "\n".join(str(item) for item in value)
        else:
            serialized_value = str(value)
        token_count = count_tokens(serialized_value)
        sections[key] = {
            "content": serialized_value,
            "tokens": token_count
        }
        total_tokens += token_count

    if total_tokens <= max_tokens:
        return context

    section_order = ["PubChem Data", "Wikipedia Articles", "arXiv Articles", "PubMed Articles"]
    truncated_context = {"Drug Name": context["Drug Name"]}
    remaining_tokens = max_tokens - count_tokens(context["Drug Name"])

    for section in section_order:
        if section in sections:
            if sections[section]["tokens"] <= remaining_tokens:
                truncated_context[section] = context[section]
                remaining_tokens -= sections[section]["tokens"]
            else:
                truncated_content = truncate_text(sections[section]["content"], remaining_tokens, tokenizer)
                truncated_context[section] = truncated_content
                break

    return truncated_context

def truncate_text(text, max_tokens, tokenizer):
    """Truncate a string to fit within a given token limit."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)



