import logging

logger = logging.getLogger(__name__)


def expand_query(query: str):

    expansions = [
        query,
        f"{query} method",
        f"{query} technique",
        f"{query} approach"
    ]

    logger.info(f"Generated {len(expansions)} expanded queries")

    return expansions