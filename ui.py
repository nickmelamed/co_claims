import streamlit as st
import requests
import json
import os
from typing import List, Dict
import pandas as pd # to display dummy data

# Configuration
API_URL = "http://rag-service:8000"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Page config
st.set_page_config(
   page_title="CoClaims AI", # Updated title
   page_icon="🤖",
   layout="wide"
)

# add dark mode
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        min-width: 500px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 500px !important;
        min-width: 500px !important;
    }

    /* Text + headers */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #ffffff !important;
    }

    /* Inputs */
    .stTextInput input {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 10px;
    }

    /* Buttons */
    .stButton button {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)


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


def trigger_ingestion(bucket: str = "co-claims-scraped-data", prefix: str = "mdna_facts_v2_first100.csv"):
   """Trigger document ingestion."""
   try:
       response = requests.post(
           f"{st.session_state.api_url}/ingest",
           headers=get_headers(),
           json={"bucket": bucket, "prefix": prefix},
           timeout=None
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
  
#    # Query settings
#    st.subheader("Query Settings")
#    top_k = st.slider("Number of sources (top_k)", min_value=1, max_value=10, value=5)
#    temperature = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
  
#    st.divider()
  
#    # Ingestion
#    st.subheader("📥 Data Ingestion")
#    with st.expander("Ingest Documents"):
#        bucket = st.text_input("S3 Bucket", value="co-claims-scraped-data")
#        prefix = st.text_input("S3 Prefix", value="mdna_facts_v2_first100.csv")
      
#        if st.button("Start Ingestion"):
#            with st.spinner("Ingesting documents..."):
#                result = trigger_ingestion(bucket, prefix)
#                if "error" in result:
#                    st.error(f"Error: {result['error']}")
#                else:
#                    stats = result.get("statistics") or {}
#                    files_n = stats.get("files_processed", 0)
#                    chunks_n = stats.get("total_chunks", 0)
#                    st.success(f"✅ Ingested {files_n} files, {chunks_n} chunks")
#                    errs = stats.get("errors") or []
#                    if errs:
#                        st.warning("Some objects failed:\n" + "\n".join(errs))
  
#    st.divider()
  
   # Clear conversation
   if st.button("🗑️ Clear Conversation"):
       st.session_state.messages = []
       st.rerun()
  
   st.divider()
  
   # Info
   st.caption("💡 Tip: Clear your conversation before entering a prompt about a new claim.")
   st.caption("📚 Referenced sources are shown at the bottom with the results of each prompt!")
   st.caption("PLACEHOLDER - ADD LINK TO METRIC DEFINITIONS DOCUMENT")


# Main chat interface
#st.title("CoClaims AI") # Updated title
#logo to match website
st.markdown("""
<div style="display: flex; align-items: center; gap: 24px;">
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        width: 180px;
        height: 180px;
        background-color: #1e293b;
        border: 3px solid rgba(59,130,246,0.3);
        border-radius: 9999px;
    ">
        <svg xmlns="http://www.w3.org/2000/svg" width="90" height="90"
            viewBox="0 0 24 24" fill="none" stroke="#f59e0b"
            stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="m8 11 2 2 4-4"></path>
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.3-4.3"></path>
        </svg>
    </div>
    <span style="font-size: 72px; font-weight: 700;">CoClaims AI</span>
</div>
""", unsafe_allow_html=True)
st.caption("Hey there, I'm CoClaims, your personal AI engine for evaluating the claims made by public companies. Input a prompt with a question about a company claim. Then see the outputted evaluation metrics and reference sources!  **Note that nothing displayed here constitutes legal or financial advice.**") # Updated

# Display chat messages
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])
      
       # Show sources if available
       if "sources" in message and message["sources"]:
           with st.expander("📚 View Sources"):
               for i, source in enumerate(message["sources"], 1):
                   st.markdown(f"""
**Source {i}**
- **File**: `{source.get('s3_key', '')}`
- **Relevance Score**: {source['score']:.3f}
- **Chunk**: {source.get('chunk_index', '')}
- **Fact Type**: {source.get('fact_type', '—')}
- **Filing Date**: {source.get('filing_date', '—') or '—'}
- **Source URL**: {source.get('source_url', '—') or '—'}
                   """)

# Chat input
if prompt := st.chat_input("Ask your question here! Example: Has NVIDIA engaged in circular financing since 2025?"):
   # Add user message to chat
   st.session_state.messages.append({"role": "user", "content": prompt})
  
   with st.chat_message("user"):
       st.markdown(prompt)
  
   # Get AI response
   with st.chat_message("assistant"):
       with st.spinner("Thinking..."):
           ## COMMENTED OUT SECTION TO INSERT DUMMY UI DATA (START)
#             result = call_chat_api(prompt, top_k=top_k, temperature=temperature)
          
#             if "error" in result:
#                 response = f"❌ Error: {result['error']}"
#                 st.error(response)
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": response
#                 })
#             else:
#                 answer = result.get("answer", "No answer generated")
#                 sources = result.get("sources", [])
              
#                 st.markdown(answer)
              
#                 # Show sources
#                 if sources:
#                     with st.expander("📚 View Sources"):
#                         for i, source in enumerate(sources, 1):
#                             st.markdown(f"""
# **Source {i}**
# - **File**: `{source.get('s3_key', '')}`
# - **Relevance Score**: {source['score']:.3f}
# - **Chunk**: {source.get('chunk_index', '')}
# - **Fact Type**: {source.get('fact_type', '—')}
# - **Filing Date**: {source.get('filing_date', '—') or '—'}
# - **Source URL**: {source.get('source_url', '—') or '—'}
#                             """)
              
#                 # Add assistant message to chat
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": answer,
#                     "sources": sources
#                 })
## COMMENTED OUT SECTION TO INSERT DUMMY UI DATA (END)

        # START DUMMY UI INPUT 

        # --- DUMMY RESPONSE TEXT ---
            answer = "This is a placeholder response for demo purposes."

            st.markdown(answer)

            # --- DUMMY METRICS UI ---
            st.subheader(f"Claim Analysis: {prompt}")

            metrics = {
                "Evidence Support Score (ESS)": 0.71,
                "Evidence Contradictory Score (ECS)": 0.42,
                "Evidence Availability Score (EAS)": 0.83,
                "Claim Specificity Score (CSS)": 0.64,
                "Claim Testability Score (CTS)": 0.58,
                "Evidence Recency Score (ERS)": 0.76,
                "Source Diversity Score (SDS)": 0.69,
                "External Verifiability Score (EVS)": 0.72,
                "Claim Scope Score (CScope)": 0.55,
                "Claim Measurability Score (CMS)": 0.61,
                "Hedging Level Score (HLS)": 0.33
            }

            dashboard, chat = st.columns([3,1])

            with dashboard:

                # st.markdown("### Evidence Metrics")

                # cols = st.columns(4)
                # i = 0
                # for k, v in metrics.items():
                #     cols[i].metric(k, f"{v:.2f}")
                #     i += 1
                #     if i == 4:
                #         cols = st.columns(4)
                #         i = 0

                # TEST REPLACING METRIC DISLPAY WITH PROGRESS BARS UP TO 1

                # st.markdown("### Evidence Metrics")

                # for k, v in metrics.items():
                #     st.markdown(f"**{k}** — {v:.2f}")
                #     st.progress(v)

                #TEST DISPLAY 2
                st.markdown("### Evidence Metrics")

                cols = st.columns(4)
                i = 0
                for k, v in metrics.items():
                    with cols[i]:
                        st.markdown(f"**{k}**")
                        st.markdown(f"""
<div style="background:#374151; border-radius:6px; height:10px; width:100%;">
  <div style="background:#f59e0b; width:{v*100}%; height:10px; border-radius:6px;"></div>
</div>
<div style="font-size:12px; color:#9ca3af; margin-top:2px;">{v:.2f}</div>
""", unsafe_allow_html=True)
                    i += 1
                    if i == 4:
                        cols = st.columns(4)
                        i = 0

                st.markdown("### Supporting vs Contradictory Evidence")

                evidence_chart = pd.DataFrame({
                    "Type": ["Supporting", "Contradictory"],
                    "Count": [14, 6]
                }).set_index("Type")

                st.bar_chart(evidence_chart)

                st.markdown("### Evidence Sources")

                evidence_data = pd.DataFrame({
                    "Source Type": [
                        "GitHub Repository",
                        "Financial Filing",
                        "Tech News Article",
                        "Company Blog",
                        "Research Paper",
                        "Industry Report"
                    ],
                    "Sentiment": [
                        "Support",
                        "Contradict",
                        "Neutral",
                        "Support",
                        "Support",
                        "Contradict"
                    ],
                    "Evidence Strength": [3,2,1,2,3,2],
                    "Date": [
                        "2025-11-02",
                        "2025-10-14",
                        "2025-09-08",
                        "2025-08-22",
                        "2025-07-10",
                        "2025-05-02"
                ]
            })

            st.dataframe(evidence_data, use_container_width=True)

        # --- STORE IN SESSION (so chat history still works) ---
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": []
            })

# END DUMMY UI INPUT

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
   st.caption("🔗 [API Docs](http://localhost:8000/docs)")
with col2:
   st.caption("📊 [Vector Database: Pinecone")
with col3:
   st.caption(f"💬 {len(st.session_state.messages)} messages")