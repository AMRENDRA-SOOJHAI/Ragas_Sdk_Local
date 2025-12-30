import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

api_key = os.environ["OPENAI_API_KEY"]

# RAG PIPELINE
def build_rag_chain():

    loader = TextLoader("input.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key
    )

    template = """
    You are an assistant for question answering tasks.
    Use the following retrieved context to answer the question.
    If you don't know the answer, say you don't know.
    Use two sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt| llm| StrOutputParser())
    
    return {
        "rag_chain": rag_chain,
        "retriever": retriever,
        "embeddings": embeddings,
        "llm": llm
    }


# RAGAS EVALUATION
def run_ragas():
    ragas_rag = build_rag_chain()

    rag_chain = ragas_rag["rag_chain"]
    retriever = ragas_rag["retriever"]
    embeddings = ragas_rag["embeddings"]
    llm = ragas_rag["llm"]

    print(f"rag_chain: {rag_chain}")
    print(f"retriever: {retriever}")
    print(f"embeddings: {embeddings}")
    print(f"llm: {llm}")

    questions = [
        "When did modern Artificial Intelligence research begin and what event marked its formal birth?",
        "What are the main advantages and disadvantages of Artificial Intelligence?",
        "How is Artificial Intelligence used in the banking sector?",
        "What role does data play in the success of modern AI systems?",
        "What are the upcoming major trends in Artificial Intelligence?"
    ]

    ground_truths = [
        "Modern AI research began in the 1950s, and the Dartmouth Conference in 1956 marked the formal birth of Artificial Intelligence as a field.",
        "AI offers advantages such as automation, improved productivity, accuracy, and continuous operation, but it also has disadvantages including job displacement, bias, lack of transparency, high computational cost, and ethical concerns.",
        "In banking, AI is used for fraud detection, transaction monitoring, credit scoring, loan approvals, risk assessment, chatbots, algorithmic trading, biometric security, regulatory compliance, and personalized financial services.",
        "Data is critical to modern AI systems because machine learning models rely on large, high-quality datasets to learn patterns, make predictions, and improve accuracy, while biased or poor data leads to biased outcomes.",
        "Upcoming AI trends include Generative AI, Large Language Models, multimodal AI, AI agents, autonomous systems, Edge AI, smaller efficient models, AI safety research, explainable AI, and evolving regulatory frameworks."
    ]

    answers = []
    contexts = []

    for i in questions:
        answers.append(rag_chain.invoke(i))
        docs = retriever.invoke(i)
        print(f'--------{docs}--------')
        contexts.append([d.page_content for d in docs])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )
    return result


# =========================
# testing code 
# =========================
# if __name__ == "__main__":
#     rag = build_rag_chain()
#     print(rag["rag_chain"].invoke("What is Artificial Intelligence?"))

