import streamlit as st
import requests
import json
import os
from typing import List, Dict
import pandas as pd

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

if "analysis_done" not in st.session_state:
   st.session_state.analysis_done = False

if "followup_messages" not in st.session_state:
   st.session_state.followup_messages = []

if "followup_count" not in st.session_state:
   st.session_state.followup_count = 0


def get_headers():
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


def call_chat_api(query: str, top_k: int = 5, temperature: float = 0.7):
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
            return {"error": f"{response.status_code} - {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def check_health():
   """Check if the RAG service is available."""
   try:
       response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
       return response.status_code == 200
   except:
       return False

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
  
   # Clear conversation
   if st.button("🗑️ Clear Conversation", use_container_width=True):
       st.session_state.messages = []
       st.session_state.analysis_done = False
       st.session_state.followup_messages = []
       st.session_state.followup_count = 0
       st.rerun()
  
   st.divider()
  
   # Info
   st.caption("💡 Tip: Clear your conversation before entering a prompt about a new claim.")
   st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
   st.caption("📚 Referenced sources are shown at the bottom with the results of each prompt!")
   st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
   st.caption("PLACEHOLDER - ADD LINK TO METRIC DEFINITIONS DOCUMENT")


# Main chat interface
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

st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
st.caption("Hey there, I'm CoClaims, your personal AI engine for evaluating the claims made by public companies. Input a prompt with a question about a company claim. Then see the outputted evaluation metrics and reference sources!  **Note that nothing displayed here constitutes legal or financial advice.**")
st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)


if not st.session_state.analysis_done:
    if prompt := st.chat_input("Ask your question here!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.analysis_done = True
        st.session_state.current_prompt = prompt
        st.rerun()


# analysis + dashboard
if st.session_state.analysis_done:

    # Run analysis only once
    if "analysis_result" not in st.session_state:
        with st.spinner("Running evaluation..."):
            result = call_chat_api(st.session_state.current_prompt)

            if "error" in result:
                st.error(result["error"])
                st.stop()

            st.session_state.analysis_result = result

    result = st.session_state.analysis_result
    prompt = st.session_state.current_prompt

    overview = result.get("overview", "")
    metrics = result.get("metrics", {})
    credibility = result.get("credibility", 0.0)
    evidence_counts = result.get("evidence_counts", {})
    sources = result.get("sources", [])

    # main chat output 
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        st.markdown(overview)
        st.subheader(f"Claim Analysis: {prompt}")

        dashboard, chat = st.columns([3,1])

        # credibility circle 
        with chat:
            filled = int(credibility * 100)
            empty = 100 - filled

            st.markdown(f"""
<div style="display:flex; flex-direction:column; align-items:center;">
 <svg width="180" height="180" viewBox="0 0 36 36">
   <path d="M18 2 a 16 16 0 1 1 0 32 a 16 16 0 1 1 0 -32"
         fill="none" stroke="#374151" stroke-width="3.5"/>
   <path d="M18 2 a 16 16 0 1 1 0 32 a 16 16 0 1 1 0 -32"
         fill="none" stroke="#f59e0b" stroke-width="3.5"
         stroke-dasharray="{filled} {empty}"
         transform="rotate(-90 18 18)"/>
   <text x="18" y="20.5" text-anchor="middle"
         fill="#ffffff" font-size="7" font-weight="bold">
         {credibility:.2f}
   </text>
 </svg>
 <div style="color:#9ca3af;">Credibility</div>
</div>
""", unsafe_allow_html=True)

        # metrics dashboard
        with dashboard:

            # Split metrics into groups
            evidence_metrics = ["ESS", "ECS", "EAS", "ERS", "ESTS", "EAGS", "SRS", "EVS"]
            claim_metrics = ["CMS", "LCS", "HLS", "CScope"]

            def render_metric_group(title, keys):
                st.markdown(f"### {title}")
                cols = st.columns(4)
                i = 0

                for k in keys:
                    if k not in metrics:
                        continue

                    val = float(metrics[k]) if isinstance(metrics[k], (int, float)) else 0

                    with cols[i]:
                        st.markdown(f"**{k}**")
                        st.markdown(f"""
<div style="background:#374151; border-radius:6px;">
 <div style="background:#f59e0b; width:{val*100}%; height:10px;"></div>
</div>
<div style="font-size:12px;">{val:.2f}</div>
""", unsafe_allow_html=True)

                    i += 1
                    if i == 4:
                        cols = st.columns(4)
                        i = 0

            render_metric_group("Evidence Quality", evidence_metrics)
            render_metric_group("Claim Quality", claim_metrics)

            # evidence counts
            st.markdown("### Supporting vs Contradictory Evidence")

            support = evidence_counts.get("supporting", 0)
            contradict = evidence_counts.get("contradicting", 0)

            chart_df = pd.DataFrame({
                "Type": ["Supporting", "Contradictory"],
                "Count": [support, contradict]
            }).set_index("Type")

            st.bar_chart(chart_df)

            # sources
            st.markdown("### Evidence Sources")

            if sources:
                df = pd.DataFrame([
                    {
                        "Source": s.get("s3_key"),
                        "Score": round(s.get("score", 0), 3),
                        "Chunk": s.get("chunk_index"),
                        "Timestamp": s.get("timestamp")
                    }
                    for s in sources
                ])

                st.dataframe(df, use_container_width=True)

                # Expandable evidence drill-down
                st.markdown("#### 🔍 Inspect Evidence")

                for i, s in enumerate(sources):
                    with st.expander(f"{s.get('s3_key')} (score {s.get('score', 0):.2f})"):
                        st.write(s.get("text", "No text available"))

            else:
                st.info("No sources returned.")

    # follow up chat without 
    st.divider()
    st.markdown("#### 💬 Follow-up Chat")
    st.caption("Ask follow-ups about the analysis above.")

    for msg in st.session_state.followup_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if followup := st.chat_input("Ask a follow-up question..."):

        st.session_state.followup_messages.append({
            "role": "user",
            "content": followup
        })

        # Lightweight follow-up (no metrics re-render)
        followup_result = call_chat_api(f"{prompt}\nFollow-up: {followup}")
        reply = followup_result.get("overview", "No response generated.")

        st.session_state.followup_messages.append({
            "role": "assistant",
            "content": reply
        })

        st.rerun()

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
   st.caption("🔗 [API Docs](http://localhost:8000/docs)")
with col2:
   st.caption("📊 [Vector Database: Pinecone")
with col3:
   st.caption(f"💬 {len(st.session_state.messages)} messages")