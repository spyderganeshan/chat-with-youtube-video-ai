import streamlit as st
from model import get_transcript,get_chunk_text,create_faiss_index,search_faiss_index,ask_deepseek


# Streamlit UI
st.title("🎥 YouTube Video Q&A with AI")
st.write("Enter a YouTube video URL and ask questions based on its transcript.")

# Input for YouTube URL
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Extract Transcript"):
    if video_url:
        transcript = get_transcript(video_url)
        if transcript:
            st.success("Transcript extracted successfully!")
            chunks = get_chunk_text(transcript)
            index, stored_chunks = create_faiss_index(chunks)
            st.session_state["index"] = index
            st.session_state["chunks"] = stored_chunks
            st.session_state["ready"] = True
        else:
            st.error("Failed to fetch transcript.")
    else:
        st.warning("Please enter a valid YouTube URL.")
        
# Question input
if "ready" in st.session_state:
    question = st.text_input("Ask a question based on the video:")
    if st.button("Get Answer"):
        index = st.session_state["index"]
        chunks = st.session_state["chunks"]
        relevant_chunks = search_faiss_index(question, index, chunks)
        answer = ask_deepseek(question, relevant_chunks)
        st.write("### Answer:")
        st.write(answer)