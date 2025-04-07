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
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = "".join([entry["text"] for entry in transcript])
        return full_text
    except Exception as e:
        return str(e)
    
def get_chunk_text(text, max_length=500):
    """
    Split a text into chunks of maximum length.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Function to create FAISS vector DB
def create_faiss_index(chunks):
    """Creates a FAISS index from transcript chunks"""
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks

# Function to search for relevant context
def search_faiss_index(question, index, chunks, k=3):
    """Finds the top-k most relevant chunks from FAISS"""
    question_embedding = embed_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k)
    return [chunks[i] for i in indices[0]]

def ask_deepseek(question, relevant_chunks):
    """Uses DeepSeek model via OpenRouter to answer based on video transcript"""
    try:
        # Prepare the context
        context = ' '.join(relevant_chunks)
        
        prompt = f"""
        You are an AI assistant. Answer the following question **only** using the provided transcript context.

        Context:
        {context}

        Question: {question}

        If the answer is not in the context, say: "Sorry, I couldn't find the answer in the video transcript."
        """

        # Make the API call
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "YouTube QA App",
            },
            model="deepseek/deepseek-v3-base:free",  # Using the documented model ID
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on YouTube video transcripts."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and return the response
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error details: {str(e)}")  # Print full error for debugging
        return f"Sorry, I encountered an error while processing your request. Please try again later."