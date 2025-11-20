from typing import List

import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Path to the FAISS database used for vector storage
DB_FAISS_PATH = "vectorstore/db_faiss"

# Template for the final QA agent that grounds responses in retrieved context.
custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Query refiner prompt encourages the model to clarify Camino-specific questions.
query_refiner_template = """
You are the Camino Query Refiner Agent. Rewrite the user's Camino de Santiago question so it is
short, unambiguous, and ready for document retrieval. Keep locations, distances, dates, and
logistics. If the question is already clear, return it unchanged.

Original question: {question}

Refined question:
"""

# Document coach prompt extracts factual nuggets and sources from retrieved chunks.
document_coach_template = """
You are the Camino Document Coach Agent. Given retrieved Camino notes and the original user
question, list the top factual takeaways that answer the question. Each bullet must cite its
source using [source: value] pulled from the provided context metadata. Mention if information is
missing.

Context:
{context}

Question: {question}

Coach notes:
"""

# Lazy-loaded singletons keep the workshop experience fast by avoiding repeated downloads.
_EMBEDDINGS = None
_VECTOR_STORE = None
_RETRIEVER = None
_LLM = None
_QA_CHAIN = None
_QUERY_REFINER = None
_DOCUMENT_COACH = None


def set_custom_prompt():
    """
    Provides the Retrieval QA agent with a transparent prompt that reinforces grounded answers.
    """
    return PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )


def retrieval_qa_chain(llm, prompt, db):
    """
    Creates a retrieval-based QA chain using a language model, a prompt, and a database retriever.

    Args:
        llm: The language model to be used for generating answers.
        prompt: The prompt template to guide the model's responses.
        db: The database object used to retrieve relevant documents based on the query.

    Returns:
        A configured RetrievalQA object that combines document retrieval with language model-based answering.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # Explore top-3 passages.
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def get_embeddings():
    """Load embeddings once so every user shares the same lightweight model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _EMBEDDINGS


def get_vector_store():
    """Load the FAISS index from disk and reuse it across sessions."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = FAISS.load_local(
            DB_FAISS_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _VECTOR_STORE


def get_retriever():
    """Shared retriever so agents and the QA chain see the same knowledge base."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = get_vector_store().as_retriever(search_kwargs={"k": 3})
    return _RETRIEVER


def get_llm():
    """
    Loads a pre-trained language model for use in generating responses.

    Returns:
        An instance of the CTransformers class configured with the specified model and settings.
    """
    global _LLM
    if _LLM is None:
        _LLM = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5,
        )
    return _LLM


def get_query_refiner():
    """Agent that cleans up user questions before retrieval."""
    global _QUERY_REFINER
    if _QUERY_REFINER is None:
        prompt = PromptTemplate(
            template=query_refiner_template, input_variables=["question"]
        )
        _QUERY_REFINER = LLMChain(llm=get_llm(), prompt=prompt)
    return _QUERY_REFINER


def get_document_coach():
    """Agent that summarizes retrieved docs into workshop-friendly talking points."""
    global _DOCUMENT_COACH
    if _DOCUMENT_COACH is None:
        prompt = PromptTemplate(
            template=document_coach_template, input_variables=["context", "question"]
        )
        _DOCUMENT_COACH = LLMChain(llm=get_llm(), prompt=prompt)
    return _DOCUMENT_COACH


def get_qa_chain():
    """Final answer agent that combines retrieval + grounded prompting."""
    global _QA_CHAIN
    if _QA_CHAIN is None:
        qa_prompt = set_custom_prompt()
        _QA_CHAIN = retrieval_qa_chain(get_llm(), qa_prompt, get_vector_store())
    return _QA_CHAIN


def _format_documents(docs: List[Document]) -> str:
    """Flatten LangChain documents into readable snippets with metadata."""
    if not docs:
        return "No supporting documents were retrieved."
    formatted_chunks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source") or doc.metadata.get("file_path") or "Unknown source"
        page = doc.metadata.get("page")
        location = f"{source}"
        if page is not None:
            # Pages are zero-indexed in metadata, so show a human-readable number.
            location += f" (page {page + 1})"
        snippet = doc.page_content.strip().replace("\n", " ")
        formatted_chunks.append(f"[Snippet {idx} | {location}] {snippet}")
    return "\n\n".join(formatted_chunks)


def _format_sources(docs: List[Document]) -> str:
    """Create a short, user-friendly source list for the final response."""
    if not docs:
        return "No sources found."
    unique_sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("file_path") or "Unknown document"
        page = doc.metadata.get("page")
        label = source
        if page is not None:
            label += f" (page {page + 1})"
        if label not in seen:
            seen.add(label)
            unique_sources.append(f"- {label}")
    return "\n".join(unique_sources)


@cl.on_chat_start
async def start():
    """
    Pre-load the QA chain so workshop attendees are not waiting on model downloads.
    """
    chain = get_qa_chain()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Camino Santiago Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Runs the three mini agents (query refiner, document coach, QA) so learners can observe the full
    reasoning pipeline before the final grounded answer is streamed back to them.
    """
    chain = cl.user_session.get("chain")
    user_query = (message.content or "").strip()
    if not user_query:
        await cl.Message(content="Please enter a question about the Camino de Santiago.").send()
        return

    # Agent 1: Query Refiner
    query_refiner = get_query_refiner()
    refine_runner = cl.make_async(query_refiner.run)
    refined_query = (await refine_runner({"question": user_query})).strip() or user_query
    await cl.Message(
        content=f"🧭 Query Refiner Agent rewrote your question as:\n**{refined_query}**"
    ).send()

    # Retrieve docs explicitly so the Document Coach can discuss them with the audience.
    retriever = get_retriever()
    retriever_runner = cl.make_async(retriever.get_relevant_documents)
    documents = await retriever_runner(refined_query)

    # Agent 2: Document Coach
    document_coach = get_document_coach()
    coach_runner = cl.make_async(document_coach.run)
    document_notes = (
        await coach_runner(
            {
                "context": _format_documents(documents),
                "question": user_query,
            }
        )
    ).strip()
    await cl.Message(
        content="📚 Document Coach Agent found these grounded notes:\n" + document_notes
    ).send()

    # Agent 3: Retrieval QA
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall({"query": refined_query}, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    answer += "\n\nSources:\n" + _format_sources(sources)

    await cl.Message(content=answer).send()
