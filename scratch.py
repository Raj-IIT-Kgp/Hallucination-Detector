from retrieval.web_search import WebRetriever

retriever = WebRetriever()
claim = "The combat soldier received the Medal of Honor at age 19."
queries = ['To Hell and Back (film)', 'Audie Murphy Medal of Honor age']
print(retriever.search(queries, claim, k=3))
