import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_model = None


async def get_reranker():

    global _reranker_model

    if _reranker_model is None:

        logger.info("Loading reranker model")

        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

    return _reranker_model


async def rerank_documents(query, documents, top_k=5):

    try:

        model = await get_reranker()

        pairs = [[query, doc.page_content] for doc in documents]

        scores = model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        # sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # extract only documents
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]

        for doc, score in scored_docs[:top_k]:
           print(f"\nScore: {score}")
           print(doc.page_content[:200])

        logger.info(f"Reranked {len(documents)} documents")

        return reranked_docs

    except Exception as e:

        logger.error(f"Reranking failed: {str(e)}")
        raise