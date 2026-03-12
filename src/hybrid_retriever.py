import logging
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self,vector_store, bm25_retriver):
        """
        Initialize Hybrid Retriever with both FAISS and BM25
        """
        self.vector_store= vector_store
        self.bm25_retriver=bm25_retriver


    def search(self,query:str, k:int = 5):

        try:

            #semantic search
            vector_results= self.vector_store.similarity_search(query,k=k)

            #Keyword search
            bm25_results= self.bm25_retriver.search(query,k=k)

            logger.info(
                f"Vector returned {len(vector_results)} docs, BM25 returned {len(bm25_results)} docs"
            )

            #merge result
            combined_results= vector_results+bm25_results

            #remove duplicats
            unique_docs=[]
            seen=set()

            for doc in combined_results:

                if doc.page_content not in seen:
                    unique_docs.append(doc)
                    seen.add(doc.page_content)

            logger.info(f"Hybrid retrieval returned {len(unique_docs)} docs")

            return unique_docs[:k]
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise
        


