import streamlit as st
import requests
import json
import os
from typing import List, Dict
import pandas as pd # Added to display dummy data

# Configuration
API_URL = "http://rag-service:8000"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Page config
st.set_page_config(
   page_title="CoClaims AI", # Updated title
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
           timeout=120
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


def trigger_ingestion(bucket: str = "co-claims-scraped-data", prefix: str = "mdna_facts_v2_first100.csv"):
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
       bucket = st.text_input("S3 Bucket", value="co-claims-scraped-data")
       prefix = st.text_input("S3 Prefix", value="mdna_facts_v2_first100.csv")
      
       if st.button("Start Ingestion"):
           with st.spinner("Ingesting documents..."):
               result = trigger_ingestion(bucket, prefix)
               if "error" in result:
                   st.error(f"Error: {result['error']}")
               else:
                   stats = result.get("statistics") or {}
                   files_n = stats.get("files_processed", 0)
                   chunks_n = stats.get("total_chunks", 0)
                   st.success(f"✅ Ingested {files_n} files, {chunks_n} chunks")
                   errs = stats.get("errors") or []
                   if errs:
                       st.warning("Some objects failed:\n" + "\n".join(errs))
  
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
st.title("CoClaims AI") # Updated title
st.caption("Ask a question about a company claim. Then see the outputted evaluation metrics and follow up with me about the results if needed!") # Updated

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
           # COMMENTED OUT BLOCK REPLACED TEMPORARILY WITH DUMMY UI DATA (START)
        #    result = call_chat_api(prompt, top_k=top_k, temperature=temperature)
          
        #    if "error" in result:
        #        response = f"❌ Error: {result['error']}"
        #        st.error(response)
        #        st.session_state.messages.append({
        #            "role": "assistant",
        #            "content": response
        #        })
        #    else:
        #        answer = result.get("answer", "No answer generated")
        #        sources = result.get("sources", [])
              
        #        st.markdown(answer)
              
        #        # Show sources
        #        if sources:
        #            with st.expander("📚 View Sources"):
        #                for i, source in enumerate(sources, 1):
        #                    st.markdown(f"""
        #                    **Source {i}**: `{source['file']}` 
        #                    - Relevance Score: {source['score']:.3f} 
        #                    - Chunk: {source['chunk_index']}
        #                    """)
              
        #        # Add assistant message to chat
        #        st.session_state.messages.append({
        #            "role": "assistant",
        #            "content": answer,
        #            "sources": sources
        #        })
        # COMMENTED OUT BLOCK REPLACED TEMPORARILY WITH DUMMY UI DATA (END)

                # --- REAL API CALL ---
        result = call_chat_api(prompt, top_k=top_k, temperature=temperature)

        if "error" in result:
            response = f"❌ Error: {result['error']}"
            st.error(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        else:
            overview = result.get("overview", "No overview generated")
            metrics = result.get("metrics", {})
            credibility = result.get("credibility", 0.0)
            evidence_counts = result.get("evidence_counts", {})
            sources = result.get("sources", [])

            # --- OVERVIEW TEXT ---
            st.markdown(overview)

            st.subheader(f"Claim Analysis: {prompt}")

            dashboard, chat = st.columns([3,1])

            with dashboard:

                # -------------------------
                # METRICS (DYNAMIC)
                # -------------------------
                st.markdown("### Evidence Metrics")

                cols = st.columns(4)
                i = 0

                for k, v in metrics.items():
                    try:
                        cols[i].metric(k, f"{float(v):.2f}")
                    except:
                        cols[i].metric(k, str(v))

                    i += 1
                    if i == 4:
                        cols = st.columns(4)
                        i = 0

                # -------------------------
                # CREDIBILITY SCORE
                # -------------------------
                st.markdown("### Overall Credibility")
                st.metric("Credibility Score", f"{credibility:.2f}")

                # -------------------------
                # EVIDENCE COUNTS (REAL)
                # -------------------------
                st.markdown("### Supporting vs Contradictory Evidence")

                support = evidence_counts.get("supporting", 0)
                contradict = evidence_counts.get("contradicting", 0)

                evidence_chart = pd.DataFrame({
                    "Type": ["Supporting", "Contradictory"],
                    "Count": [support, contradict]
                }).set_index("Type")

                st.bar_chart(evidence_chart)

                # -------------------------
                # SOURCES TABLE (REAL)
                # -------------------------
                st.markdown("### Evidence Sources")

                if sources:
                    evidence_data = pd.DataFrame([
                        {
                            "Source": s.get("file"),
                            "Relevance Score": round(s.get("score", 0), 3),
                            "Chunk": s.get("chunk_index"),
                            "Timestamp": s.get("timestamp")
                        }
                        for s in sources
                    ])

                    st.dataframe(evidence_data, use_container_width=True)
                else:
                    st.info("No sources returned.")

            # --- STORE IN SESSION ---
            st.session_state.messages.append({
                "role": "assistant",
                "content": overview,
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
