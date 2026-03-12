import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import asyncio
import logging
from vector_store import get_vector_store
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from reranker import rerank_documents
from query_expander import expand_query
from generator import generate_answer
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]= os.getenv("groq_api-key")
groq_api_key= os.getenv("groq_api-key")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH= "data/documents"
VECTOR_DB_PATH = "vector_store"


async def load_document():
    """
    Loadd all documents from DATA_PATH

    """
    try:
      docs=[]
      file_paths=[]

      for file in Path(DATA_PATH).glob("*.pdf"):
          loader= PyPDFLoader(str(file))
          docs.extend(loader.load())
          file_paths.append(str(file))
      
      logger.info(f"Loaded {len(docs)} pages from {len(file_paths)} files")
      return docs,file_paths
    
      
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

async def slipt_documents(documents):
    """
    Split documents in smaller chunks.
    """
    try:

      text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
      chunks= text_splitter.split_documents(documents)
      logger.info(f"Created {len(chunks)} chunks")
      return chunks
    except Exception as e:
       logger.error(f"Chunking failed: {str(e)}")
       raise


async def main():
   
   try:
      #load all the pdf docs
      docs,file_paths = await load_document()

      # Chuck all the loaded documents
      chunks= await slipt_documents(docs)
    
      
      # Embeding & store all the chunks in vector database 
      vector_db = await get_vector_store(chunks,file_paths)
      logger.info("Vector DB ready")
      
      # create BM25 retriever
      bm25 = BM25Retriever(chunks)


      # create hybrid retriever
      hybrid= HybridRetriever(vector_db,bm25)

      query = "agricultural field boundary detection"

      #Expand query
      expanded_queries = expand_query(query)
      
      all_expanded_queries=[]

      for q in expanded_queries:
         quires=hybrid.search(q,k=10)
         all_expanded_queries.extend(quires)
      
      #Remove duplicate chunks
      uniq_retrive_docs=[]
      seen=set()

      for q in all_expanded_queries:
         if q.page_content not in seen:
            uniq_retrive_docs.append(q)
            seen.add(q.page_content)
      logger.info(f"Retrieved {len(uniq_retrive_docs)} unique documents")

   
      reranked_docs = await rerank_documents(query, uniq_retrive_docs, top_k=5)
      answer = generate_answer(
         query,
         reranked_docs,
         groq_api_key,
         model_name="llama-3.1-8b-instant"
      )

      logger.info("\n==============================")
      logger.info("ANSWER")
      logger.info("==============================\n")
      logger.info(answer)
    
   except Exception as e:
      logger.error(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
