import hashlib


def hash_file(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_dataset_hash(file_hashes):
    """
    Generate a deterministic hash for a set of file hashes.
    """
    combined = "".join(sorted(file_hashes))
    return hashlib.md5(combined.encode()).hexdigest()