import warnings
import logging
import os
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models
from sentence_transformers import SentenceTransformer
from os import listdir, getenv
from os.path import join, isfile
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import warnings
import torch
import logging
from functools import wraps
import os

# Custom warning filter
def filter_torch_class_warnings(record):
    return not record.getMessage().startswith('Examining the path of torch.classes')

# Configure logging with the custom filter
logging.getLogger().addFilter(filter_torch_class_warnings)

# Suppress specific PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Create a context manager to suppress specific warnings
class SuppressWarnings:
    def __enter__(self):
        self.catch_warnings = warnings.catch_warnings()
        self.catch_warnings.__enter__()
        warnings.filterwarnings('ignore', message='.*torch.classes.*')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.catch_warnings.__exit__(exc_type, exc_val, exc_tb)

# Decorator for functions that might raise the warning
def suppress_torch_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with SuppressWarnings():
            return func(*args, **kwargs)
    return wrapper

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Change to WARNING to reduce verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Suppress tensorflow warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class API_Chatbot:
    @suppress_torch_warnings
    def __init__(self, data_dir: str = "scraped_data_new_gpt"):
        """Initialize the chatbot with minimal logging output."""
        # Temporarily suppress stdout during model loading
        with open(os.devnull, 'w') as devnull:
            old_stdout = os.dup(1)
            os.dup2(devnull.fileno(), 1)
            try:
                load_dotenv()
                self._validate_env_vars()
                
                # Initialize encoder quietly
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
                
                # Initialize Qdrant client
                self.qdrant = QdrantClient(
                    url=getenv("QDRANT_URL"),
                    api_key=getenv("QDRANT_API_KEY")
                )
                self.qdrant.set_model(
                    self.qdrant.DEFAULT_EMBEDDING_MODEL,
                    providers=["CPUExecutionProvider"]
                )

                self.docs = self.prepare_docs(data_dir)
                self.save_to_db()
                
                genai.configure(api_key=getenv('GEMINI_API_KEY'))
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            finally:
                # Restore stdout
                os.dup2(old_stdout, 1)
                os.close(old_stdout)
            pass

    def _validate_env_vars(self):
        required_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'GEMINI_API_KEY']
        missing_vars = [var for var in required_vars if not getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def prepare_docs(self, dir_name: str) -> List[Dict[str, str]]:
        if not isfile(dir_name):
            files = [join(dir_name, f) for f in listdir(dir_name) 
                    if isfile(join(dir_name, f))]
        else:
            files = [dir_name]
            
        docs = []
        for file_path in files:
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                    docs.append({
                        "name": file_path,
                        "content": content,
                    })
            except Exception as e:
                continue
                
        return docs

    def save_to_db(self):
        if not self.qdrant.collection_exists("crustdata"):
            self.qdrant.create_collection(
                collection_name="crustdata",
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            )
            
            points = [
                models.PointStruct(
                    id=idx,
                    vector=self.encoder.encode(doc["content"]).tolist(),
                    payload=doc
                )
                for idx, doc in enumerate(self.docs)
            ]
            self.qdrant.upload_points(
                collection_name="crustdata",
                points=points
            )
    @suppress_torch_warnings
    def get_context(self, query: str, limit: int = 3) -> str:
        hits = self.qdrant.query_points(
            collection_name="crustdata",
            query=self.encoder.encode(query).tolist(),
            limit=limit
        ).points
        
        if not hits:
            return ""
            
        return hits[0].payload['content']
        pass

    def rewrite_query(self, query: str) -> str:
        """Rewrite the query to focus on API endpoints."""
        prompt = """
        Rewrite this user query to focus on finding API endpoints for people search.
        
        Examples:
        User: "how do i find mechanical engineers in sf"
        Rewrite: "API endpoint for searching people by job title mechanical engineer and location San Francisco"
        
        User: "How do I search for people by their title and company?"
        Rewrite: "API endpoint for searching people by current job title and current company"
        
        Now rewrite this query:
        {query}
        """.format(query=query)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return query

    def call_llm(self, query: str) -> str:
        """Generate a response with API endpoint details."""
        try:
            rewritten = self.rewrite_query(query)
            context = self.get_context(rewritten)
            
            prompt = """
            You are a helpful API documentation expert. Based on the user's query and the documentation context, provide a clear response about how to use the People Search API.

            If the query is about finding people, always include these details:
            1. The exact API endpoint
            2. The HTTP method (GET/POST)
            3. Required parameters
            4. Optional parameters
            5. A brief example

            For example, if someone asks about finding engineers in a location, respond like this:

            To search for people based on their job title and location, use:

            Endpoint: POST /v2/people-search
            
            Required Parameters:
            - title: Job title to search for (e.g., "mechanical engineer")
            - location: Location to search in (e.g., "San Francisco")
            
            Optional Parameters:
            - distance: Search radius in miles (default: 25)
            - current_only: Only show current positions (default: true)
            - limit: Maximum number of results (default: 20)
            
            Example Request:
            {
                "title": "mechanical engineer",
                "location": "San Francisco",
                "current_only": true,
                "distance": 25
            }

            Context: {context}
            Query: {query}
            """.format(context=context, query=query)
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            # Provide a more helpful error message
            return """I apologize, but I encountered an error processing your request. Let me provide a general answer."""
