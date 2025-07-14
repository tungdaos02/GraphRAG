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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os,re

load_dotenv()
class Entities(BaseModel):
    name: list[str] = Field(..., description="All the person, organization, or business entities that appear in the text.")

class GraphRetriever():

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    embed = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=os.getenv("OPENAI_API_KEY"))

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

    def extracting_entities_question(self, question):
        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        """Dựa vào câu hỏi bạn hãy trích xuất các entities có trong câu hỏi với entities là người, tổ chức,ngày,giờ , địa điểm hoặc khái niệm,... và nếu có entity là địa điểm hoặc thời gian thì tổng hợp lại thêm entity là entitySumLocation với entity địa điểm và entitySumDay với entity là ngày.
                        Ví dụ mẫu:
                        1.Câu hỏi: "Lịch trình tour du lịch Hà Nội - Đà Nẵng 4 ngày 3 đêm". List entities: {{"tour du lịch", "Hà Nội", "Đà Nẵng", "4 ngày 3 đêm", "hanoidanang", "4n3d"}} Với "hanoidanang" chính là entitySumLocation của các entitties địa điểm :["Hà Nội", "Đà Nẵng"] và "4n3d" là entitySumDay của entity "4 ngày 3 đêm" Bạn chỉ trả lời {{ "tour du lịch", "Hà Nội", "Đà Nẵng", "4 ngày 3 đêm", "hanoidanang", "4n3d"}}
                        2.Câu hỏi: "tôi muốn đi du lịch biển 2 ngày". List entities: {{"du lịch biển", "2 ngày 1 đêm", "2n1d", "2n2d"}} Với "2n1d" và "2n2d" là entitySumDay của entity "2 ngày" vì số đêm luôn nhỏ hơn ngày là 1 đơn vị nên nếu chỉ có dữ liệu là 2 ngày thì ngầm hiểu là 1 đêm nên sẽ có entity là "2n1d" và "2n2d" Bạn chỉ trả lời {{ "du lịch biển", "2 ngày" , "2n1d", "2n2d"}}
                        3.Câu hỏi: "Tour du lịch biển HCM - Đà nẵng". List entities: {{"tour du lịch biển", "HCM" , "Đà Nắng", "hcmdanang"}} Với "hcmdanang" là entitySumLocation của entitíe: ["HCM", "Đà Nắng"] Bạn chỉ được trả lời {{"tour du lịch biển", "HCM" , "Đà Nắng", "hcmdanang"}} 
                        4.Câu hỏi: "Cho tôi chi tiết lịch trình tour du lịch bắt đầu tại HCM rồi di chuyển đến Huế và kết thúc tại Đà Nẵng 4 ngày". List entities:{{"tour du lịch", "HCM", "Huế", "Đà Nẵng", "hcmhuedanang", "4 ngày", "4n3d", "4n4d"}} Với "hcmhuedanang" chính là entitySumLocation của các entitties địa điểm :["HCM", "Huế", "Đà Nẵng"], còn "4n3d" và "4n4d" là entitySumDay của entity "4 ngày" Bạn chỉ trả lời {{ "tour du lịch", "HCM", "Huế", "Đà Nẵng", "hcmhuedanang", "4 ngày", "4n3d", "4n4d"}}
                        Cách hoạt động trích xuất entity là ngày hiểu như nếu trích xuất được entity là "X ngày Y đêm" thì entitySumDay có dạng là "XnYd" còn nếu entity chỉ có mỗi "X ngày" thì entitySumDay sẽ có dạng là {{"XnXd" , "Xn(X-1)d"}}
                        Câu hỏi: {question}""",
                    )
                ]
            )
        entity_chain = prompt | self.llm.with_structured_output(Entities)
        entitties = entity_chain.invoke({"question": question})
        return entitties

    def filter_chunk_by_question(self, question):
        entities = self.extracting_entities_question(question)
        chunk_fil_time = []
        for entity in entities.name:
            response = self.graph.query(
                """
                MATCH (d:Document)
                WHERE d.time = $time
                RETURN d;
                """,
                {
                    "time": entity
                }
            )
            chunk_fil_time += response
        chunk_fil_time_and_location = []
        if not chunk_fil_time:
            for entity in entities.name:
                response = self.graph.query(
                    """
                    MATCH (d:Document)
                    WHERE d.source = $source
                    RETURN d;
                    """,
                    {
                        "source": entity
                    }
                )
                chunk_fil_time_and_location += response
        else:
            for entity in entities.name:
                filtered = [item for item in chunk_fil_time if item['d'].get('source') == entity]
                chunk_fil_time_and_location += filtered
        return chunk_fil_time_and_location

    def embed_text(self,text: str) -> list:
        return self.embed.embed_query(text)

    def similarity_search(self, question, k=5):
        q_emb = np.array(self.embed_text(question)).reshape(1, -1)
        docs = self.filter_chunk_by_question(question)
        doc_embs = np.array([d['d']['embedding'] for d in docs])
        sims = cosine_similarity(q_emb, doc_embs)[0]
        topk_indices = sims.argsort()[-k:][::-1]
        return [docs[i]['d']['text'] for i in topk_indices]

    def graph_retriever(self, question, type_graph) -> str:
        entities = self.extracting_entities_question(question)
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
        #Graph Data
        subgraph_extend_context = self.graph_retriever(question, "extend")
        graph_data_filter = self.filter_sub_graph(question, subgraph_extend_context)

        #Vectorsearch Data
        extended_question = self.extended_question(question)
        filter_chunk = self.filter_chunk_by_question(question)
        if not filter_chunk:
            vector_retriever = self.vector_index.as_retriever(search_kwargs={'k': 5})
            vector_data = [el.page_content for el in vector_retriever.invoke(extended_question)]
        else:
            vector_data = self.similarity_search(question)
        final_data = f"""Graph data:
            {graph_data_filter}
            Vector data:
            {"#Document ".join(vector_data)}
            """
        print(final_data) 
        return final_data
    
    def run_chain(self,question: str) -> str:
        template = """Answers the question based only on the following context and for general questions that are not clear, ask the user for more details:
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
