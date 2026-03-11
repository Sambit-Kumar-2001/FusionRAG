from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import asyncio
import logging
from vector_store import create_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH= "data/documents"

async def load_document():
    """
    Loadd all documents from DATA_PATH

    """
    try:
      docs=[]

      for file in Path(DATA_PATH).glob("**/*.pdf"):
          loader= PyPDFLoader(str(file))
          docs.extend(loader.load())
          return docs
          
        #   logger.info(f"Loaded documents: {len(docs)}")
    
      
    
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
      docs = await load_document()
    #   logger.info(f"loadded documents:{len(docs)}")

      # Chuck all the loaded documents
      chunks= await slipt_documents(docs)
    #   logger.info(f"Created {len(chunks)} Chunks")
    #   logger.info("\nSample chunk:\n")
    #   logger.info(chunks[0].page_content[:50])
      
      # Embeding & store all the chunks in vector database 
      vector_db = await create_vector_store(chunks)
      logger.info("Vector DB ready")


      query = "How can satellite imagery detect agricultural fields?"
      results = vector_db.similarity_search(query, k=3)

    #   logger.info("="*100,"Results Start", "="*100)

    #   logger.info(f"results:\n {results}")
    #   logger.info("="*100,"Results End", "="*100)

      for doc in results:
            print("\n---- Result ----\n")
            print(doc.page_content[:300])
    
   except Exception as e:
      logger.error(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
