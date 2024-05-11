# =========================================================
# Created by Nguyen Quoc Thai
# Date: 08/05/2024
# Description:
#   This script demonstrates how to create a FAISS vector store from documents stored in CSV files.
#   It loads document data from CSV files located in a specified directory, computes embeddings using the Hugging Face Transformers library,
#   and then creates a FAISS vector store from these embeddings.
# =========================================================

import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, DirectoryLoader

def create_parser():
    """Parser for main function

    Returns:
        args: all arguments for main function.
    """
    parser = argparse.ArgumentParser(description='Parser for data extraction and save to database')
    parser.add_argument('--data_path', type=str, help='Path to folder contains csv file', default='./data')
    parser.add_argument('--data_faiss_path', type=str, help='Path to folder store database', default='vectorstore/db_faiss')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_parser()
    DATA_PATH = args.data_path
    DB_FAISS_PATH = args.data_faiss_path

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})

    loader = DirectoryLoader(DATA_PATH,
                                glob='*.csv',
                                loader_cls=CSVLoader)

    documents = loader.load()

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)

