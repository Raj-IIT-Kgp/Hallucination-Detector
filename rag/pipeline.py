from retrieval.vector_store import VectorStore
from llm.generator import generate_response


class RAGPipeline:

    def __init__(self, documents):

        self.store = VectorStore()

        self.store.add_documents(documents)


    def answer(self, question):

        documents = self.store.search(
            question,
            k=3
        )


        context = "\n".join(documents)


        prompt = f"""
You are a helpful assistant.

Answer the question only using the provided context.

Context:
{context}


Question:
{question}


Answer:
"""


        response = generate_response(prompt)


        return response