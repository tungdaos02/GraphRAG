from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
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

    def graph_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
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
       
        clean_line = []
        entities = entity_chain.invoke({"question": question})
        for entity in entities.name:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node AS startNode, score
                    MATCH (startNode)-[r1]-(n1)
                    WHERE NOT type(r1) = 'MENTIONS'
                    WITH startNode, n1, r1
                    OPTIONAL MATCH (n1)-[r2]-(n2)
                    WHERE NOT type(r2) = 'MENTIONS' AND n2 <> startNode
                    RETURN DISTINCT
                    startNode.id AS from,
                    type(r1) AS rel1,
                    n1.id AS middle,
                    type(r2) AS rel2,
                    n2.id AS to
                    LIMIT 100;
                """,
                {"query": entity},
            )
        for item in response:
            from_part = item['from']
            rel1 = item['rel1']
            middle = item['middle']
            rel2 = item.get('rel2')
            to = item.get('to')

            if rel2 and to:
                line = f"{from_part} {rel1} {middle} {rel2} {to}"
            else:
                line = f"{from_part} {rel1} {middle}"
            line = re.sub(r"_", " ", line)
            clean_line.append(line)
        result = "\n".join(clean_line)
        return result
    
    def full_retriever(self, question: str) -> str:
        graph_data = self.graph_retriever(question)
        vector_retriever = self.vector_index.as_retriever()
        vector_data = [el.page_content for el in vector_retriever.invoke(question)]

        final_data = f"""Graph data:
            {graph_data}
            Vector data:
            {"#Document ".join(vector_data)}
            """ 
        print(final_data)
        print("======================================")
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
