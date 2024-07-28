import os
from glob import glob

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF
from tqdm import tqdm


def add_text_to_database(input_doc: str, doc_meta_data: Dict[str, str], chunk_size=1024, chunk_overlap=256,
                         chroma: Chroma = None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_text(input_doc)
    chroma.add_texts(texts=splits, metadatas=[doc_meta_data for _ in range(len(splits))])


def extract_text_from_pdf(pdf_path):
    documents = []
    file_name = os.path.basename(pdf_path)
    with fitz.open(pdf_path) as reader:
        for page_num in tqdm(range(reader.page_count), f"Extracting pages from {file_name}"):
            text = reader.load_page(page_num).get_text()
            metadata = {
                "file_name": file_name,
                "file_path": pdf_path,
                "page_number": page_num + 1  # Page numbers are 1-indexed
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def process_pdfs_from_directory(inp_directory, chroma):
    files = glob(f'{inp_directory}/*.pdf')
    for file in tqdm(files, "processing files"):
        document = extract_text_from_pdf(file)
        for page in tqdm(document, f"indexing pages"):
            add_text_to_database(page.page_content, page.metadata, chunk_size=1024, chunk_overlap=768, chroma=chroma)


def main():
    chroma = Chroma(collection_name="example_collection",
                    embedding_function=HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                                             model_kwargs={'trust_remote_code': True}),
                    persist_directory=r"/tmp/example_collection")
    process_pdfs_from_directory('/home/ganesh/Documents/pd/RAG/src_txt', chroma)


if __name__ == '__main__':
    main()
