import streamlit as st
import requests
import json
import os
from typing import List, Dict

# Configuration
API_URL = "http://rag-service:8000"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Page config
st.set_page_config(
   page_title="RAG AI Assistant",
   page_icon="🤖",
   layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
   st.session_state.messages = []

if "api_url" not in st.session_state:
   st.session_state.api_url = API_URL


def get_headers():
   """Get headers with auth token if available."""
   headers = {"Content-Type": "application/json"}
   if AUTH_TOKEN:
       headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
   return headers


def check_health():
   """Check if the RAG service is available."""
   try:
       response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
       return response.status_code == 200
   except:
       return False


def call_chat_api(query: str, top_k: int = 5, temperature: float = 0.7):
   """Call the RAG service chat endpoint."""
   try:
       response = requests.post(
           f"{st.session_state.api_url}/chat",
           headers=get_headers(),
           json={
               "query": query,
               "top_k": top_k,
               "temperature": temperature
           },
           timeout=30
       )
      
       if response.status_code == 200:
           return response.json()
       else:
           return {"error": f"API error: {response.status_code} - {response.text}"}
  
   except requests.exceptions.Timeout:
       return {"error": "Request timeout - the service took too long to respond"}
   except requests.exceptions.ConnectionError:
       return {"error": "Connection error - could not reach the RAG service"}
   except Exception as e:
       return {"error": f"Unexpected error: {str(e)}"}


def trigger_ingestion(bucket: str = "capstone-data", prefix: str = "rag-files/"):
   """Trigger document ingestion."""
   try:
       response = requests.post(
           f"{st.session_state.api_url}/ingest",
           headers=get_headers(),
           json={"bucket": bucket, "prefix": prefix},
           timeout=300
       )
       return response.json()
   except Exception as e:
       return {"error": str(e)}


# Sidebar
with st.sidebar:
   st.title("⚙️ Settings")
  
   # Service status
   st.subheader("Service Status")
   if check_health():
       st.success("✅ RAG Service Connected")
   else:
       st.error("❌ RAG Service Unavailable")
  
   st.divider()
  
   # Query settings
   st.subheader("Query Settings")
   top_k = st.slider("Number of sources (top_k)", min_value=1, max_value=10, value=5)
   temperature = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
  
   st.divider()
  
   # Ingestion
   st.subheader("📥 Data Ingestion")
   with st.expander("Ingest Documents"):
       bucket = st.text_input("S3 Bucket", value="capstone-data")
       prefix = st.text_input("S3 Prefix", value="rag-files/")
      
       if st.button("Start Ingestion"):
           with st.spinner("Ingesting documents..."):
               result = trigger_ingestion(bucket, prefix)
               if "error" in result:
                   st.error(f"Error: {result['error']}")
               else:
                   st.success(f"✅ Ingested {result.get('files_processed', 0)} files, {result.get('total_chunks', 0)} chunks")
  
   st.divider()
  
   # Clear conversation
   if st.button("🗑️ Clear Conversation"):
       st.session_state.messages = []
       st.rerun()
  
   st.divider()
  
   # Info
   st.caption("💡 Tip: Ask questions about your documents!")
   st.caption("📚 Sources are shown below each answer")


# Main chat interface
st.title("🤖 RAG AI Assistant")
st.caption("Ask questions about your documents")

# Display chat messages
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])
      
       # Show sources if available
       if "sources" in message and message["sources"]:
           with st.expander("📚 View Sources"):
               for i, source in enumerate(message["sources"], 1):
                   st.markdown(f"""
                   **Source {i}**: `{source['file']}` 
                   - Relevance Score: {source['score']:.3f} 
                   - Chunk: {source['chunk_index']}
                   """)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
   # Add user message to chat
   st.session_state.messages.append({"role": "user", "content": prompt})
  
   with st.chat_message("user"):
       st.markdown(prompt)
  
   # Get AI response
   with st.chat_message("assistant"):
       with st.spinner("Thinking..."):
           result = call_chat_api(prompt, top_k=top_k, temperature=temperature)
          
           if "error" in result:
               response = f"❌ Error: {result['error']}"
               st.error(response)
               st.session_state.messages.append({
                   "role": "assistant",
                   "content": response
               })
           else:
               answer = result.get("answer", "No answer generated")
               sources = result.get("sources", [])
              
               st.markdown(answer)
              
               # Show sources
               if sources:
                   with st.expander("📚 View Sources"):
                       for i, source in enumerate(sources, 1):
                           st.markdown(f"""
                           **Source {i}**: `{source['file']}` 
                           - Relevance Score: {source['score']:.3f} 
                           - Chunk: {source['chunk_index']}
                           """)
              
               # Add assistant message to chat
               st.session_state.messages.append({
                   "role": "assistant",
                   "content": answer,
                   "sources": sources
               })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
   st.caption("🔗 [API Docs](http://localhost:8000/docs)")
with col2:
   st.caption("📊 [Vector Database: Pinecone")
with col3:
   st.caption(f"💬 {len(st.session_state.messages)} messages")
