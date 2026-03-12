import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def build_context(documents):

    context_parts = []

    for i, doc in enumerate(documents):

        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[Source {i+1}] {doc.page_content}\n(Source file: {source})"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, documents, groq_api_key, model_name="llama-3.1-8b-instant"):

    try:

        context = build_context(documents)

        llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY using the provided context. Also answer briefly that a normal layman can understood. add a example at the end of the answer

If the answer is not in the context say:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer with citations like [Source 1].
"""
        )

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "question": query
        })

        return response

    except Exception as e:

        logger.error(f"LLM generation failed: {str(e)}")
        raise