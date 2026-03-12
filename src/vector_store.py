import os
import logging
from langchain_community.vectorstores import FAISS

from embeddings import get_embedding_model
from utils.generate_dataset_hash import hash_file, get_dataset_hash
from utils.metadata_manager import load_metadata, save_metadata

logger = logging.getLogger(__name__)

VECTOR_ROOT = "vector_store"
METADATA_PATH = os.path.join(VECTOR_ROOT, "metadata.json")

os.makedirs(VECTOR_ROOT, exist_ok=True)


async def get_vector_store(chunks, file_paths):

    try:
        # Generate hashes for all files
        file_hashes = {path: hash_file(path) for path in file_paths}
        
        # Generate a unique hash for this specific set of documents
        dataset_hash = get_dataset_hash(list(file_hashes.values()))
        
        # Define a unique directory for this dataset
        dataset_dir = os.path.join(VECTOR_ROOT, f"index_{dataset_hash}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        index_path = os.path.join(dataset_dir, "index.faiss")

        # Check if this specific index already exists
        if os.path.exists(index_path):
            logger.info(f"Existing index found for dataset {dataset_hash} — loading")
            embeddings = await get_embedding_model()
            vector_store = FAISS.load_local(
                dataset_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store

        else:
            logger.info(f"No existing index for dataset {dataset_hash} — creating new FAISS index")
            embeddings = await get_embedding_model()
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            # Save the index in the unique directory
            vector_store.save_local(dataset_dir)
            
            # Save metadata inside the same directory
            metadata = {"files": file_hashes, "dataset_hash": dataset_hash}
            save_metadata(metadata_path, metadata)

            return vector_store

    except Exception as e:
        logger.error(f"Vector store error: {str(e)}")
        raise