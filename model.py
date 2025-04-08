from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
# set initial api key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
)
# load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_transcript(video_url):
    """
    Get the transcript of a YouTube video using its video ID.
    """
    # use openai wisper

# Function to create FAISS vector DB
def create_faiss_index(chunks):
    """Creates a FAISS index from transcript chunks"""
    # Euse huggingface to make vector db
    return 


def ask_deepseek(question, relevant_chunks):
        # use huggingfce openai to get answer or other models
        return 