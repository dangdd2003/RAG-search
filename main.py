import os
import argparse
import yaml
from source.utils import test_cuda, get_sample_docs
from source.main_llm import query
from source.datasource import add_document, get_document, pop_database

# Currently only use GPU 2 and 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def main():
    parser = argparse.ArgumentParser(description="RAG search")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU.")
    parser.add_argument(
        "--cuda",
        metavar="NUMBER ARRAY",
        action="store",
        help="Use specific GPU device. Eg: [--cuda 0,1,2] for using GPU 0, 1, and 2.",
    )
    parser.add_argument(
        "--test-cuda", action="store_true", help="Test weather CUDA is available."
    )
    parser.add_argument(
        "--model", metavar="STRING", action="store", help="Model name to use."
    )
    parser.add_argument(
        "--torch-dtype",
        metavar="STRING",
        action="store",
        default="auto",
        help="Torch data type to use (precesion). Eg: float32, bfloat16, float16.",
    )
    parser.add_argument(
        "--max-new-tokens",
        metavar="NUMBER",
        action="store",
        default=2048,
        help="Maximum number of output tokens to generate.",
    )
    parser.add_argument(
        "--get-samples",
        default=False,
        action="store_true",
        help="Install sample documents.",
    )
    parser.add_argument(
        "--add-docs",
        action="store_true",
        help="Add document in defined path to Chroma database.",
    )
    parser.add_argument(
        "--get-docs",
        type=str,
        metavar="STRING",
        action="store",
        help="Query for documents in Chroma database.",
    )
    parser.add_argument(
        "--pop-db",
        action="store_true",
        help="Pop all documents in Chroma database.",
    )
    parser.add_argument(
        "--query-llm",
        type=str,
        metavar="STRING",
        action="store",
        help="Ask the model with a prompt.",
    )
    parser.add_argument(
        "--search",
        type=str,
        metavar="STRING",
        action="store",
        help="Use LLM for searching provided documents.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Whether to use reranker cross-encoder for better searching results.",
    )

    args = parser.parse_args()

    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU.")

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print(f"Using GPU: {args.cuda}")

    if args.test_cuda:
        test_cuda()

    if args.model:
        model_name = args.model
        print(f"Using user defined model: {model_name}")
    else:
        model_name = config["models"]["llm"]["name"]
        print(f"Using default model: {model_name}")

    if args.torch_dtype:
        torch_dtype = args.torch_dtype
        print(f"Using user defined main llm's precesion: {torch_dtype}")
    else:
        torch_dtype = config["models"]["llm"]["torch-dtype"]
        print(f"Using default main llm's precesion: {torch_dtype}")

    if args.get_samples:
        print("Installing sample documents...")
        get_sample_docs(config["data"][0]["documents-dir"])
        print("Finished installing sample documents.")

    if args.no_reranker:
        print("Not using reranker! This will reduce time but may reduce accuracy.")

    if args.query_llm:
        messages = [{"role": "user", "content": args.query_llm}]
        print("Querying directly LLM model...")
        print("--------------------------------------------\nAnswer: ")
        query(messages, model_name, torch_dtype, args.max_new_tokens)

    if args.search:
        print("Using default embedding: ", config["models"]["embeddings"]["name"])
        if not args.no_reranker:
            print("Using default SBERT model: ", config["models"]["sbert"]["name"])
        source, context = get_document(
            args.search,
            config["chroma"]["database-dir"],
            config["models"]["embeddings"]["name"],
            not args.no_reranker,
            config["models"]["sbert"]["name"],
            k=50,
            top_n=10,
        )
        messages_en = [
            {
                "role": "user",
                "content": f"""Answer the question based only on the following context:
{context}

---------------------------------------------------------------
Now, based on the context above, answer the following question:
{args.search}
""",
            }
        ]
        print("Searching in provided documents...")
        print("--------------------------------------------\nAnswer: ")
        query(messages_en, model_name, torch_dtype, args.max_new_tokens)
        print("--------------------------------------------\nSources: ", source)

    if args.add_docs:
        print("Adding document in defined path to Chroma database...")
        print("Using default embedding: ", config["models"]["embeddings"]["name"])
        add_document(
            embedding_name=config["models"]["embeddings"]["name"],
            documents_dir=config["data"][0]["documents-dir"],
            database_dir=config["chroma"]["database-dir"],
            chunk_size=config["data"][0]["chunk-size"],
            chunk_separator=config["data"][0]["chunk-separators"],
        )

    if args.get_docs:
        print("Querying documents in Chroma database...")
        print("Using default embedding: ", config["models"]["embeddings"]["name"])
        if not args.no_reranker:
            print("Using default SBERT model: ", config["models"]["sbert"]["name"])
        source, context = get_document(
            args.get_docs,
            config["chroma"]["database-dir"],
            config["models"]["embeddings"]["name"],
            not args.no_reranker,
            config["models"]["sbert"]["name"],
            k=50,
            top_n=10,
        )
        print(
            f"Contents: {context}\n--------------------------------------------\nSources: {source}"
        )

    if args.pop_db:
        pop_database(config["chroma"]["database-dir"])
        print("Popped all documents in Chroma database.")


if __name__ == "__main__":
    main()
