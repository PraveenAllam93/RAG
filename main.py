import faiss
import numpy as np 
import os
from openai import OpenAI
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tiktoken
from scipy.spatial.distance import cosine
from langchain.load import dumps, load
import json
import chromadb
from fastapi import FastAPI, Depends, Request, Cookie
from fastapi.params import Body
from dependencies import get_timestamp, set_llm_config, format_docs, encode_documents, get_collection_snow, get_collection_others
from log import setup_logging, get_app_logger, get_error_logger

setup_logging()
app_logger = get_app_logger()
error_logger = get_error_logger()

app = FastAPI(
    title='RAG - Apollo',
    description='LLM Bot, which has access to the private data',
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    contact={
        "name": "Praveen Allam",
        "email": "saipraveen.allam@copart.com"
    }
)

@app.get("/")
async def root(timestamp: str = Depends(get_timestamp)):
    app_logger.info(f"Accessed / @ {timestamp}")
    return {"message": f"Hello world! {timestamp}"} 


@app.get("/chatbot")
async def chatbot(query: str, timestamp: str = Depends(get_timestamp)):
    base_llm = False
    app_logger.info(f"Accessed chatbot / @ {timestamp}")
    llm = set_llm_config()
    snow_collection, others = get_collection_snow(), get_collection_others()
    try: 
        if not snow_collection or not others:
            base_llm = True
            app_logger.info("No collections found, using Base LLM to answer")
            try:
                response = llm.invoke(query)
                app_logger.info(f"LLM provided a result for the user query!!!!")
            except AuthenticationError:
                error_logger.error("Authentication failed. Invalid API key!")
                raise
            except APIConnectionError:
                error_logger.error("Network error: Unable to connect to the OpenAI API!")
                raise
            except RateLimitError:
                error_logger.error("Rate Limit exceeded: Too many requests!")
                raise
            except OpenAIError as e:
                error_logger.error(f"OpenAI API error: {e}")
                raise
            except response.Timeout:
                error_logger.error("Timeout error: The request took too long to complete.")
                raise
            except Exception as e:
                error_logger.error(f"An unexpected error occurred: {type(e).__name__, str(e)}")
                raise
            
        else:
            results = snow_collection.query(
                        query_texts=query, # Chroma will embed this for you
                        n_results=2 # how many results to return
                    )
            relevant_docs = format_docs(results["documents"])
            app_logger.info(f"Related Documents are retrieved amd formated!!!!")
            print(relevant_docs)
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {e}")
        raise
    if base_llm == False:    
        template = """Answer the question based only on the following retrieved chunks:
        {retrieved_chunks}
        
        Question : {query}
        """
        prompt = ChatPromptTemplate.from_template(template)
        filled_prompt = prompt.format(retrieved_chunks = relevant_docs, query = query)
        app_logger.info("prompt is filled and sending it to llm.............!!")
        print(filled_prompt)
        try: 
            response = llm.invoke(filled_prompt)
            app_logger.info(f"LLM provided a result for the user query!!!!")
        except AuthenticationError:
            error_logger.error("Authentication failed. Invalid API key!")
            raise
        except APIConnectionError:
            error_logger.error("Network error: Unable to connect to the OpenAI API!")
            raise
        except RateLimitError:
            error_logger.error("Rate Limit exceeded: Too many requests!")
            raise
        except OpenAIError as e:
            error_logger.error(f"OpenAI API error: {e}")
            raise
        except response.Timeout:
            error_logger.error("Timeout error: The request took too long to complete.")
            raise
        except Exception as e:
            error_logger.error(f"An unexpected error occurred: {type(e).__name__, str(e)}")
            raise
    return str(response.content)


@app.post("/post_data")
async def post_data(directory: dict = Body(...)):
    # {"directory" : "path"}
    try:
        app_logger.info("encoding documents are being called.....!!!")
        encode_documents(directory["directory"])
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {type(e).__name__, str(e)}")
    return {"status" : 200, "message" : "files in the given directory are uploaded"}    
    
