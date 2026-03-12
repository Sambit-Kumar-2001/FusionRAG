from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

_embedding_model = None


async def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        logger.info("Loading embedding model (first time only)")

        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return _embedding_model