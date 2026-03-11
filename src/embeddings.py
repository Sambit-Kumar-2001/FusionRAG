from langchain_huggingface import HuggingFaceEmbeddings

import logging

logger = logging.getLogger(__name__)

async def get_embedding_model():
    """
    Load embedding model from HuggingFace

    """
    try:
        model_name="BAAI/bge-small-en-v1.5"
        embeddings=HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device":"cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        logger.info("Embedding model loaded successfully")

        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        raise

