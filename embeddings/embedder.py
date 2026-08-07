from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embedding(text):

    vector = model.encode(text)

    return vector