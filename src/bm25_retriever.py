import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self,chunks):
        """
        Build a BM25 index from document chunks.
        """
        try:
            self.documents=chunks
            # tokenizer corpus
            self.corpus=[doc.page_content.split() for doc in chunks]
            self.bm25 = BM25Okapi(self.corpus)

            logger.info("BM25 index created successfully")
        except Exception as e:
            logger.error(f"BM25 initialization failed: {str(e)}")
            raise
    def search(self,query:str,k:int=5):
        """
        Retrieve top-k documents using BM25.
        """

        try:
            tokenize_query= query.lower().split()
            scores= self.bm25.get_scores(tokenize_query)
            
            ranked_indices= sorted(
                range(len(scores)),
                key=lambda i:scores[i],
                reverse=True

            )[:k]

            return [self.documents[i] for i in ranked_indices]
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            raise

