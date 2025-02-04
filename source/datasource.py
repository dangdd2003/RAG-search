import os
import shutil
import torch
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SimilarityFunction
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker


def load_embedding(name: str) -> None:
    return HuggingFaceEmbeddings(
        model_name=name,
        multi_process=True,
        model_kwargs={
            "similarity_fn_name": SimilarityFunction.EUCLIDEAN,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )


def add_document(
    embedding_name: str,
    documents_dir: str,
    database_dir: str,
    chunk_size: int,
    chunk_separator: list[str],
) -> None:
    embedding = load_embedding(embedding_name)
    if not os.path.exists(documents_dir):
        os.mkdir(documents_dir)
    if not os.path.exists(database_dir):
        os.mkdir(database_dir)

    documents = PyPDFDirectoryLoader(documents_dir).load()
    chunks = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embedding_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size / 10,
        strip_whitespace=True,
        separators=chunk_separator,
    ).split_documents(documents)

    db = Chroma(
        collection_name="documents",
        persist_directory=database_dir,
        embedding_function=embedding,
    )
    last_source_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        current_source_id = f"{source}:{page}"
        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id
        chunk.metadata["chunk_id"] = chunk_id

    existed_chunks = db.get(include=[])
    existed_ids = set(existed_chunks["ids"])
    print(f"Number of existed chunks in Chroma DB: {len(existed_ids)}")
    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existed_ids
    ]
    print(f"Number of new chunks to add: {len(new_chunks)}")
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new chunks to Chroma DB...")
        new_chunks_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_chunks_ids, show_progress=True)
            print("Finished adding new chunks to Chroma DB.")
        except Exception as e:
            print(f"Error while adding new chunks to Chroma DB: {e}")
            print("No new chunks added.")
    else:
        print("No new chunks to add.")


def get_document(
    query: str,
    database_dir: str,
    embedding_name: str,
    reranker: bool,
    bert_name: str,
    k: int,
    top_n: int,
) -> tuple[list[str], str]:
    db = Chroma(
        collection_name="documents",
        persist_directory=database_dir,
        embedding_function=load_embedding(embedding_name),
    )
    if reranker:
        bert = HuggingFaceCrossEncoder(
            model_name=bert_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        retrieved_chunks = db.as_retriever(search_kwargs={"k": k})
        compressor = CrossEncoderReranker(model=bert, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retrieved_chunks,
        )
        retrieved_docs = compression_retriever.invoke(query)
    else:
        retrieved_docs = db.similarity_search(query, k)
    source = [doc.metadata.get("chunk_id", None) for doc in retrieved_docs]
    context = "".join([doc.page_content for doc in retrieved_docs])
    return source, context


def pop_database(database_dir: str) -> None:
    shutil.rmtree(database_dir)
