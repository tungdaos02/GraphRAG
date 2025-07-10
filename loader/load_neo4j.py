from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import Document
# from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

def load_data_neo4j(raw_text: str):

    graph = Neo4jGraph(
            url = os.getenv("NEO4J_URI"),                
            username = os.getenv("NEO4J_USERNAME"),          
            password = os.getenv("NEO4J_PASSWORD"),         
            database = os.getenv("NEO4J_DATABASE")
        ) 
    docs = [Document(page_content=raw_text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    llm_transformer = LLMGraphTransformer(
        llm = llm,
        additional_instructions=(
        "Please ensure that all entities, relationship, label and values are "
        "extracted and returned in Vietnamese, preserving the original language "
        "of the input text."),
        allowed_nodes=["Người", "Vị trí", "Địa điểm", "Khái niệm", "Sự kiện", 
                       "Tổ chức", "Hành động", "Thời gian", "Ngày", "Đối tượng",
                       "Đồ vật", "Sản phẩm", "Dịch vụ", "Tài nguyên", "Tài liệu",
                       "Món ăn", "Thực phẩm", "Vật phẩm", "Số liệu", "Dân tộc", ],
    )

    grahp_documents = llm_transformer.convert_to_graph_documents(chunks)

    graph.add_graph_documents(
        grahp_documents,
        baseEntityLabel=True,
        include_source=True
    )

    graph.query(
        """
        CREATE FULLTEXT INDEX entity
        IF NOT EXISTS
        FOR (e:__Entity__)
        ON EACH [e.id]
        """
    )