from langchain_community.vectorstores import FAISS
import logging

from embeddings import get_embedding_model

logger =logging.getLogger(__name__)


async def create_vector_store(chunks):

    """
    Convert document chunks into embeddings and store in FAISS
    """

    try:
        embeddings = await get_embedding_model()

        vector_store= FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        logger.info("Vector store created Sucessfully !!!")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise

