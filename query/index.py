from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os,re

load_dotenv()
class Entities(BaseModel):
    name: list[str] = Field(..., description="All the person, organization, or business entities that appear in the text.")

class GraphRetriever():

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    embed = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vector_index = Neo4jVector.from_existing_graph(
        url      = os.getenv("NEO4J_URI"),                
        username = os.getenv("NEO4J_USERNAME"),          
        password = os.getenv("NEO4J_PASSWORD"),         
        database = os.getenv("NEO4J_DATABASE"),
        search_type = "hybrid",
        node_label = "Document",
        embedding = embed,
        text_node_properties = ["text"],
        embedding_node_property = "embedding"
    )

    graph = Neo4jGraph(
                url      = os.getenv("NEO4J_URI"),                
                username = os.getenv("NEO4J_USERNAME"),          
                password = os.getenv("NEO4J_PASSWORD"),         
                database = os.getenv("NEO4J_DATABASE")
            )    
    
    def clean_sub_graph(self, sub_graph: str) -> str:
        pattern = re.compile(r"->|[-_]")
        cleaned_lines = [
            re.sub(r"\s+", " ", pattern.sub(" ", el["output"])).strip()
            for el in sub_graph
        ]
        result = "\n".join(cleaned_lines)
        return result

    def cypher_deep_node(self, node: str):
        response = self.graph.query(
            """
            MATCH path = (startNode {id: $startName})-[rels*2]-(endNode)
                WHERE ALL(r IN rels WHERE type(r) <> 'MENTIONS')
                AND ALL(n IN nodes(path) WHERE SINGLE(m IN nodes(path) WHERE m = n))
                WITH path, nodes(path) AS nodeList, relationships(path) AS relList
                RETURN 
                    reduce(
                        s = nodeList[0].id, 
                        i IN RANGE(0, size(relList)-1) | 
                        s + " - " + type(relList[i]) + " -> " + nodeList[i+1].id
                    ) AS output
                LIMIT 200
            """,
            {
                "startName": node
            }
        )
        return response
    
    def cypher_adjacent_node(self, node: str):
        response = self.graph.query(
            """
            MATCH path = (startNode {id: $startName})-[rels*1]-(endNode)
                WHERE ALL(r IN rels WHERE type(r) <> 'MENTIONS')
                AND ALL(n IN nodes(path) WHERE SINGLE(m IN nodes(path) WHERE m = n))
                WITH path, nodes(path) AS nodeList, relationships(path) AS relList
                RETURN 
                    reduce(
                        s = nodeList[0].id, 
                        i IN RANGE(0, size(relList)-1) | 
                        s + " - " + type(relList[i]) + " -> " + nodeList[i+1].id
                    ) AS output
                LIMIT 200
            """,
            {
                "startName": node
            }
        )
        return response

    def filter_sub_graph(self, question: str,sub_grap_data) -> str:
        template = PromptTemplate(
            input_variables=["sub_grap_data", "question"],
            template= """Tôi có danh sách kết quả ngữ cảnh
                    {sub_grap_data}
                    Tôi muốn lọc ra danh sách ngữ cảnh mới phù hợp với ngữ cảnh câu hỏi:{question}. 
                    Chỉ trả về kết quả danh sách không trả lời gì thêm."""
        )
        chain = template | self.llm | StrOutputParser()
        response = chain.invoke({
            "sub_grap_data": sub_grap_data,
            "question": question
        })
        return response
    
    def graph_retriever(self, question: str,type_graph) -> str:
        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are extracting all the person, organization, event, location, times or action entities from the text.",
                    ),
                    (
                        "human",
                        "Use the given format to extract information from the following"
                        "input: {question}",
                    ),
                ]
            )
        entity_chain = prompt | self.llm.with_structured_output(Entities)
    
        entities = entity_chain.invoke({"question": question})
        result = []
        for entity in entities.name:
            if type_graph == "sub":
                response = self.cypher_adjacent_node(entity)
            elif type_graph == "extend":
                response = self.cypher_deep_node(entity)
            result += response
        result = self.clean_sub_graph(result)
        return result
    
    def extended_question(self, question: str) -> str:
        subgraph_context = self.graph_retriever(question, "sub")
        print(subgraph_context)
        sub_graph_data_filter = self.filter_sub_graph(question, subgraph_context) 
        template = PromptTemplate(
            input_variables=["sub_graph_data_filter", "question"],
            template="""Hãy viết lại câu hỏi:{question} thành câu hỏi mới cụ thể, đầy đủ và chuẩn hóa
                    bằng cách lấy thêm ngữ cảnh từ {sub_graph_data_filter}.Chỉ trả về câu hỏi mới."""
        )
        chain = template | self.llm | StrOutputParser()
        response = chain.invoke({
            "sub_graph_data_filter": sub_graph_data_filter,
            "question": question
        })
        return response
    
    def full_retriever(self, question: str) -> str:
        subgraph_extend_context = self.graph_retriever(question, "extend")
        graph_data_filter = self.filter_sub_graph(question, subgraph_extend_context)
        extended_question = self.extended_question(question)
        vector_retriever = self.vector_index.as_retriever()
        vector_data = [el.page_content for el in vector_retriever.invoke(extended_question)]

        final_data = f"""Graph data:
            {graph_data_filter}
            Vector data:
            {"#Document ".join(vector_data)}
            """ 
        return final_data
    
    def run_chain(self,question: str) -> str:
        template = """Answers the question based only on the following context:
            {context}
            Question: {question}
            Use Vietnamese to answer the question.
            Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": lambda _:self.full_retriever(question),
                "question": RunnablePassthrough(),
            }
        | prompt
        | self.llm
        | StrOutputParser()
    )
        return chain.invoke({"question": question})
