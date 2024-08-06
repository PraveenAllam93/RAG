import os
from datetime import datetime
from dotenv import load_dotenv
from log import get_app_logger, get_error_logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AuthenticationError, APIConnectionError, RateLimitError, OpenAIError
import faiss
import numpy as np 
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import tiktoken
import json
import chromadb
from pydantic import BaseModel
from pathlib import Path


app_logger = get_app_logger()
error_logger = get_error_logger()

load_dotenv()
API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

try:
    client = chromadb.PersistentClient(path = "/Users/praveenallam/Desktop/RAG")
    app_logger.info("Chroma client is configured!!")
except Exception as e:
    error_logger.error(f"An Unexpected error : {e}")
    raise

def set_llm_config():
    try:
        llm = ChatOpenAI(api_key = API_KEY, model = MODEL, temperature = 0)
        app_logger.info("llm is defined!!!")
        return llm
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {e}")
        raise
    
def create_collections():
    try: 
        snow_collection = client.get_or_create_collection(name = "snow_application")
        others = client.get_or_create_collection(name = "others")
        app_logger.info("Chroma client is set, and collections are configured!!")
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {e}")
        raise

def get_collection_snow():
    try:
        collection = client.get_collection("snow_application")
        app_logger.info("Collection snow found!!")
        return collection
    except ValueError:
        app_logger.warning(f"No collection found with 'snow' and continuing with it!!!")
        return None
    except Exception as e:
        error_logger.error(f"Unexpected error occurred while getting 'snow': {e}")
        raise
    
def get_collection_others():
    try:
        collection = client.get_collection("others")
        app_logger.info("Collection others found!!")
        return collection
    except ValueError:
        app_logger.warning(f"No collection found with 'others' and continuing with it!!!")
        return None
    except Exception as e:
        error_logger.error(f"Unexpected error occurred while getting 'others': {e}")
        raise

def read_text_files(directory):
    documents = []
    document_type = {}
    app_logger.info("started reading the files...!!!")
    for filename in os.listdir(directory):
        app_logger.info(f"{filename} is being checked!!")
        if filename.endswith(".txt"):
            app_logger.info(f"{filename} is being read!!")
            file_path = directory / filename
            with open(file_path, "r") as file:
                documents.append(file.read())
            document_type[filename] = "snow" if "snow" in filename else "not snow"
    complete_document = {}
    document_names = list(document_type.keys())
    app_logger.info("all the text files are read....!")
    itr = 0
    for document_id,document in enumerate(documents):
        document_info = {}
        document_info["name"] = document_names[itr]
        document_info["type"] = document_type[document_info["name"]]
        document_info["content"] = document
        itr += 1
        complete_document[str(document_id)] = document_info
    app_logger.info("The documents are processed compeltely")

    return complete_document

def encode_documents(directory):
    directory = Path(directory)
    create_collections()
    documents = read_text_files(directory)
    app_logger.info("started adding documents to thier respective collections....")
    snow_id, other_id = [], []
    snow_documents, other_documents = [], []
    snow_collection = get_collection_snow()
    others = get_collection_others()
    for document_id in documents.keys():
        if documents[document_id]["type"] == "snow":
            snow_id.append(document_id)
            snow_documents.append(documents[document_id]["content"])
        else:
            other_id.append(document_id)  
            other_documents.append(documents[document_id]["content"])  
    snow_collection.add(ids = snow_id, documents = snow_documents)
    others.add(ids = other_id, documents = other_documents)
    app_logger.info(f"number of collections added to snow = {snow_collection.count()}, is it correct? {snow_collection.count() == len(snow_documents)}")
    app_logger.info(f"number of collections added to others = {others.count()}, is it correct? {snow_collection.count() == len(other_documents)}")

 
def format_docs(docs):
    return "\n\n".join(info for doc in docs for info in doc)

def get_timestamp():
    return str(datetime.now())
