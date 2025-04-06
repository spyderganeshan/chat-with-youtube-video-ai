from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
# set initial api key
API_KEY=""
openai_client = openai.OpenAI(api_key=API_KEY)
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
    """Uses DeepSeek model from DeepInfra to answer based on video transcript"""
    
    prompt = f"""
    You are an AI assistant. Answer the following question **only** using the provided transcript context.

    Context:
    {' '.join(relevant_chunks)}

    Question: {question}

    If the answer is not in the context, say: "Sorry, out of syllabus."
    """

    url = "https://api.deepinfra.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-ai/deepseek-llm-7b-chat",  # or "openchat/openchat-3.5-1210"
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"
