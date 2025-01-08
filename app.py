from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,models
from sentence_transformers import SentenceTransformer
from os import listdir,getenv
from os.path import join
import google.generativeai as genai
from dotenv import load_dotenv

class API_Chatbot:
    def __init__(self):
        load_dotenv()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.qdrant = QdrantClient(url=getenv("QDRANT_URL"),api_key=getenv("QDRANT_API_KEY")) # Connect to existing Qdrant instance
        self.qdrant.set_model(self.qdrant.DEFAULT_EMBEDDING_MODEL,providers=["CPUExecutionProvider"])

        self.docs = self.prepare_docs("scraped_data_new_gpt")
        self.save_to_db()

        genai.configure(api_key=getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def prepare_docs(self,dir_name):
    
        dir = listdir(dir_name)
        files = []
        for f in dir:
            files.append(join(dir_name,f))
    
        docs = []
    
        for f in files:
            docs.append({
                "name":f,
                "content":open(f,encoding='utf-8').read(),
            })
        return docs
    
    def save_to_db(self):
    
        if not self.qdrant.collection_exists("crustdata"):
            self.qdrant.create_collection(
                collection_name="crustdata",
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                    distance=models.Distance.COSINE,
                )
            )
            self.qdrant.upload_points(
                collection_name="crustdata",
                points=[
                    models.PointStruct(
                        id=idx, vector=self.encoder.encode(doc["content"]).tolist(), payload=doc
                    )
                    for idx, doc in enumerate(self.docs)
                ],
            )
    
    
    def get_context(self,query):
        hits = self.qdrant.query_points(
            collection_name="crustdata",
            query=self.encoder.encode(query).tolist(),
            limit=3
    
        ).points
    
        return hits[0].payload['content']
    
    def rewrite_query(self,query):
        prompt = f"""
            You are an expert in query understanding and rewriting for vector databases containing API documentation. 

            Rewrite the following query so that a vector database can more effectively retrieve relevant documents:


            **Guidelines:**

            * If the original query is too conversational or specific, rewrite it as a more generic query that the vector database can better understand. 
            * For example, if the original query is "find doctors in NY", a more generic rewrite would be "give API endpoint that finds people based on filters like their job and location."
            * Output should be short, accurate and capture semantics.

            
            **New Query to rewrite:** {query}
            **Output:**

        """
    
        res = self.model.generate_content(prompt)


        return res.text

    def call_llm(self,query):
        
        rewritten = self.rewrite_query(query)
        context = self.get_context(rewritten)

        response = self.model.generate_content(f"you are a helpful assistant. Given this context : {context}.{query}")
        return response.text
    
