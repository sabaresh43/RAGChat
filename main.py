import os

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load data
    documents = SimpleDirectoryReader("data").load_data()

    # create index

    # this will convert data into vectors(embeddings) and then store as index
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist()
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


# retrieving

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
postProcessor = SimilarityPostprocessor(SimilarityPostprocessor=0.80)

# query from index
# query_engine = index.as_query_engine()
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postProcessor])


response = query_engine.query("what is Inventory Reports?")
pprint_response(response, show_source=True)
