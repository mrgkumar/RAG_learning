from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

chroma = Chroma(collection_name="example_collection",
                embedding_function=HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                                         model_kwargs={'trust_remote_code': True}),
                persist_directory=r"/tmp/example_collection")

for hit, score in chroma.similarity_search_with_score("story from mahabharata"):
    print(score)
    print(hit)
    print('=' * 20)
