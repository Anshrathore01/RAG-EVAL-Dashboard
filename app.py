import os
import shutil
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set page configuration FIRST before any other streamlit commands
st.set_page_config(
    page_title="RAG-Eval Observability Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
from src.config import (
    SUPPORTED_LLMS,
    SUPPORTED_EMBEDDINGS,
    MISTRAL_API_KEY,
    UPLOADS_DIR,
    DB_PATH
)
from src.db.sqlite import SQLiteDatabase
from src.db.models import QueryLogDTO, EvaluationScoreDTO
from src.rag.vectorstore import VectorStoreManager
from src.rag.pipeline import RAGPipeline
from src.eval.evaluator import RagasEvaluator

# Initialize session state objects
if "db" not in st.session_state:
    st.session_state.db = SQLiteDatabase()
if "vectorstore_manager" not in st.session_state:
    st.session_state.vectorstore_manager = VectorStoreManager()
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(st.session_state.vectorstore_manager)

# Check Mistral API status
api_key_valid = bool(MISTRAL_API_KEY)

# Custom premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

/* Main Fonts */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

/* Glassmorphism Panel style */
.glass-panel {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

/* Metric card specific classes */
.card-header {
    font-size: 0.85rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f3f4f6;
}
.card-value-highlight {
    color: #8b5cf6;
}
.card-subtitle {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 4px;
}

/* Badges for scores */
.badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
    text-align: center;
}
.badge-green { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-yellow { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-red { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-gray { background-color: rgba(107, 114, 128, 0.2); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.3); }

/* Custom Chat Bubbles */
.chat-user {
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 12px 16px;
    border-radius: 12px 12px 0 12px;
    margin-bottom: 15px;
    text-align: right;
    max-width: 80%;
    margin-left: auto;
}
.chat-assistant {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 12px 16px;
    border-radius: 12px 12px 12px 0;
    margin-bottom: 15px;
    max-width: 80%;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #8b5cf6;'>⚙️ RAG Configuration</h2>", unsafe_allow_html=True)

# API Status Indicator
st.sidebar.markdown("---")
if api_key_valid:
    st.sidebar.success("🔑 Mistral API Connected")
else:
    st.sidebar.error("❌ API Key Missing (check `.env`)")

st.sidebar.markdown("### Model Selection")
selected_llm = st.sidebar.selectbox("LLM Model", SUPPORTED_LLMS, index=0)
selected_embedding = st.sidebar.selectbox("Embedding Model", SUPPORTED_EMBEDDINGS, index=0)

st.sidebar.markdown("### Chunking Config")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

st.sidebar.markdown("### Retrieval Settings")
top_k = st.sidebar.slider("Retrieval Top K", min_value=1, max_value=10, value=4, step=1)

st.sidebar.markdown("### Generation Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Advanced actions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Database Operations")
if st.sidebar.button("🗑️ Reset Log Database", type="secondary"):
    st.session_state.db.clear_database()
    st.sidebar.success("Database logs cleared!")
    time.sleep(1)
    st.rerun()

# ----------------- MAIN TITLE -----------------
st.markdown("<h1 style='margin-bottom: 0px;'>🤖 RAG-Eval Observability Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; margin-top:0px;'>An observability and evaluation platform for measuring retrieval quality, answer quality, latency, token usage, and cost.</p>", unsafe_allow_html=True)

# Main tabs
tab_chat, tab_model_comp, tab_embed_comp, tab_logs, tab_analytics = st.tabs([
    "💬 Chat & Real-Time Eval",
    "📊 Model Comparison",
    "🔍 Embedding Comparison",
    "📝 Query Logs",
    "📈 System Analytics"
])

# Helper for rendering score badges
def render_score_badge(score: Optional[float]) -> str:
    if score is None:
        return "<span class='badge badge-gray'>N/A</span>"
    if score >= 0.8:
        return f"<span class='badge badge-green'>{score:.2f}</span>"
    elif score >= 0.5:
        return f"<span class='badge badge-yellow'>{score:.2f}</span>"
    else:
        return f"<span class='badge badge-red'>{score:.2f}</span>"

# ----------------- TAB 1: CHAT & REAL-TIME EVAL -----------------
with tab_chat:
    col1, col2 = st.columns([1, 1])

    # Left Column: Document Upload & Chat
    with col1:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.subheader("📁 Ingest Documents")
        uploaded_file = st.file_uploader("Upload PDF or TXT to add to the knowledge base", type=["pdf", "txt", "md"])
        
        if uploaded_file is not None:
            # Check if file has already been ingested in this session
            file_key = f"ingested_{uploaded_file.name}_{selected_embedding}"
            if file_key not in st.session_state:
                # Save file to uploads folder
                save_path = Path(UPLOADS_DIR) / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Ingesting file and generating embeddings with {selected_embedding}..."):
                    try:
                        chunks_created = st.session_state.vectorstore_manager.ingest_file(
                            file_path=save_path,
                            embedding_name=selected_embedding,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        st.session_state[file_key] = True
                        st.success(f"✓ Success! Split '{uploaded_file.name}' into {chunks_created} chunks and indexed in vectorstore.")
                    except Exception as e:
                        st.error(f"Failed to ingest file: {e}")
            else:
                st.info(f"ℹ '{uploaded_file.name}' is already ingested for embedding '{selected_embedding}'.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.subheader("💬 Ask Your RAG System")
        query_input = st.text_input("Enter your question:")
        ground_truth_input = st.text_input("Expected Answer / Ground Truth (optional, required for Context Recall):")
        
        if st.button("Run RAG Pipeline", type="primary") and query_input:
            if not api_key_valid:
                st.error("Cannot execute pipeline. Mistral API Key is missing. Please add it to your .env file.")
            else:
                with st.spinner("Retrieving contexts and generating answer..."):
                    # Run the pipeline
                    res = st.session_state.rag_pipeline.run(
                        query=query_input,
                        model_name=selected_llm,
                        embedding_name=selected_embedding,
                        top_k=top_k,
                        temperature=temperature,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        session_id="chat"
                    )
                    
                    # Store current RAG run results in session state for right column evaluation
                    st.session_state.current_run = res
                    
                    # Save to DB log
                    db_log = QueryLogDTO(
                        query=res["query"],
                        response=res["response"],
                        contexts=res["contexts"],
                        model_name=res["model_name"],
                        embedding_name=res["embedding_name"],
                        chunk_size=res["chunk_size"],
                        chunk_overlap=res["chunk_overlap"],
                        prompt_tokens=res["prompt_tokens"],
                        completion_tokens=res["completion_tokens"],
                        total_tokens=res["total_tokens"],
                        cost=res["cost"],
                        latency_ms=res["latency_ms"],
                        session_id=res["session_id"]
                    )
                    log_id = st.session_state.db.save_query_log(db_log)
                    st.session_state.current_log_id = log_id
                    st.session_state.current_ground_truth = ground_truth_input if ground_truth_input else None
                    # Clear past evaluation results
                    if "current_eval" in st.session_state:
                        del st.session_state.current_eval
                        
        # Display current chat run if it exists
        if "current_run" in st.session_state:
            run = st.session_state.current_run
            st.markdown("---")
            st.markdown(f"<div class='chat-user'>🧑 <b>Query:</b> {run['query']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-assistant'>🤖 <b>Answer:</b><br>{run['response']}</div>", unsafe_allow_html=True)
            
            # Display source chunks
            with st.expander("🔍 Retrieved Chunks / Context Sources"):
                if run["contexts"]:
                    for i, ctx in enumerate(run["contexts"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(ctx, language="markdown")
                else:
                    st.write("No chunks retrieved.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Right Column: Real-Time Observability & Evaluation
    with col2:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.subheader("⚡ Performance Observability")
        
        if "current_run" in st.session_state:
            run = st.session_state.current_run
            
            # Sub-columns for performance metrics
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='card-header'>Total Latency</div>
                    <div class='card-value card-value-highlight'>{run['latency_ms']:.0f}ms</div>
                    <div class='card-subtitle'>Retr: {run.get('retrieval_latency_ms', 0):.0f}ms | Gen: {run.get('llm_latency_ms', 0):.0f}ms</div>
                </div>
                """, unsafe_allow_html=True)
            with perf_col2:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='card-header'>Estimated Cost</div>
                    <div class='card-value card-value-highlight'>${run['cost']:.6f}</div>
                    <div class='card-subtitle'>Model: {run['model_name']}</div>
                </div>
                """, unsafe_allow_html=True)
            with perf_col3:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='card-header'>Token Usage</div>
                    <div class='card-value card-value-highlight'>{run['total_tokens']}</div>
                    <div class='card-subtitle'>In: {run['prompt_tokens']} | Out: {run['completion_tokens']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run the RAG pipeline to generate performance metrics.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.subheader("🎯 RAGAS Evaluation Scores")
        
        if "current_run" in st.session_state:
            run = st.session_state.current_run
            ground_truth = st.session_state.current_ground_truth
            
            # Show a button to trigger evaluation
            if st.button("Calculate Ragas Scores", type="secondary"):
                with st.spinner("Executing Ragas LLM-as-a-judge evaluations..."):
                    evaluator = RagasEvaluator(
                        evaluator_llm_name="mistral-large-latest", # Use large for stable evaluation
                        evaluator_embeddings_name="mistral-embed"
                    )
                    
                    scores = evaluator.evaluate_query(
                        query=run["query"],
                        response=run["response"],
                        contexts=run["contexts"],
                        ground_truth=ground_truth
                    )
                    
                    # Save to database
                    eval_dto = EvaluationScoreDTO(
                        query_id=st.session_state.current_log_id,
                        faithfulness=scores["faithfulness"],
                        answer_relevancy=scores["answer_relevancy"],
                        context_precision=scores["context_precision"],
                        context_recall=scores["context_recall"]
                    )
                    st.session_state.db.save_evaluation(eval_dto)
                    st.session_state.current_eval = scores
                    st.success("✓ Evaluation scores calculated and logged successfully!")

            # Display scores if calculated
            if "current_eval" in st.session_state:
                scores = st.session_state.current_eval
                
                # Plotly Gauge Chart for RAGAS metrics
                fig = go.Figure()
                
                metrics_list = ["Faithfulness", "Answer Relevancy", "Context Precision"]
                values_list = [
                    scores.get("faithfulness") or 0.0,
                    scores.get("answer_relevancy") or 0.0,
                    scores.get("context_precision") or 0.0
                ]
                
                if ground_truth:
                    metrics_list.append("Context Recall")
                    values_list.append(scores.get("context_recall") or 0.0)
                    
                fig.add_trace(go.Bar(
                    y=metrics_list,
                    x=values_list,
                    orientation='h',
                    marker=dict(
                        color=values_list,
                        colorscale='Viridis',
                        clim=[0.0, 1.0]
                    ),
                    text=[f"{v:.2f}" for v in values_list],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Ragas Quality Metrics",
                    xaxis=dict(title="Score", range=[0.0, 1.0]),
                    yaxis=dict(autorange="reversed"),
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Text explanations of metrics
                st.markdown("#### Metrics Analysis:")
                st.markdown(f"- **Faithfulness (Generator Quality):** {render_score_badge(scores.get('faithfulness'))} (Checks if the response is based strictly on the retrieved context.)")
                st.markdown(f"- **Answer Relevancy (Generator Quality):** {render_score_badge(scores.get('answer_relevancy'))} (Checks if the response answers the question.)")
                st.markdown(f"- **Context Precision (Retriever Quality):** {render_score_badge(scores.get('context_precision'))} (Checks if retrieved chunks are relevant to the query.)")
                if ground_truth:
                    st.markdown(f"- **Context Recall (Retriever Quality):** {render_score_badge(scores.get('context_recall'))} (Checks if the retrieved chunks align with the ground truth.)")
                else:
                    st.info("💡 Add a Ground Truth answer to enable retrieval 'Context Recall' evaluation.")
            else:
                st.info("Click the 'Calculate Ragas Scores' button to run RAGAS evaluation.")
        else:
            st.info("Run the RAG pipeline to enable evaluation.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: MODEL COMPARISON -----------------
with tab_model_comp:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Side-by-Side LLM Benchmarking")
    st.markdown("Compare Mistral Large, Medium, and Small directly for a query to benchmark latency, cost, and answer quality.")
    
    comp_query = st.text_input("Enter evaluation query for model comparison:")
    comp_ground_truth = st.text_input("Expected Answer / Ground Truth (highly recommended):")
    
    if st.button("Run Model Comparison", type="primary") and comp_query:
        if not api_key_valid:
            st.error("Cannot execute pipeline. Mistral API Key is missing.")
        else:
            models_to_test = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
            comp_results = {}
            
            with st.spinner("Running query against Mistral models and evaluating..."):
                progress_bar = st.progress(0)
                for i, m_name in enumerate(models_to_test):
                    # 1. Run RAG Pipeline
                    res = st.session_state.rag_pipeline.run(
                        query=comp_query,
                        model_name=m_name,
                        embedding_name=selected_embedding, # Keep embedding constant
                        top_k=top_k,
                        temperature=temperature,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        session_id="comparison"
                    )
                    
                    # Log run to DB
                    db_log = QueryLogDTO(
                        query=res["query"],
                        response=res["response"],
                        contexts=res["contexts"],
                        model_name=res["model_name"],
                        embedding_name=res["embedding_name"],
                        chunk_size=res["chunk_size"],
                        chunk_overlap=res["chunk_overlap"],
                        prompt_tokens=res["prompt_tokens"],
                        completion_tokens=res["completion_tokens"],
                        total_tokens=res["total_tokens"],
                        cost=res["cost"],
                        latency_ms=res["latency_ms"],
                        session_id=res["session_id"]
                    )
                    log_id = st.session_state.db.save_query_log(db_log)
                    
                    # 2. Evaluate RAGAS
                    evaluator = RagasEvaluator(evaluator_llm_name="mistral-large-latest")
                    scores = evaluator.evaluate_query(
                        query=comp_query,
                        response=res["response"],
                        contexts=res["contexts"],
                        ground_truth=comp_ground_truth if comp_ground_truth else None
                    )
                    
                    # Log evaluation to DB
                    eval_dto = EvaluationScoreDTO(
                        query_id=log_id,
                        faithfulness=scores["faithfulness"],
                        answer_relevancy=scores["answer_relevancy"],
                        context_precision=scores["context_precision"],
                        context_recall=scores["context_recall"]
                    )
                    st.session_state.db.save_evaluation(eval_dto)
                    
                    comp_results[m_name] = {
                        "run": res,
                        "scores": scores
                    }
                    progress_bar.progress((i + 1) / len(models_to_test))
                
                st.session_state.comp_results = comp_results
                st.success("Comparison runs and evaluation completed!")
                
    if "comp_results" in st.session_state:
        comp_data = st.session_state.comp_results
        
        # Display side-by-side responses in columns
        c_large, c_med, c_small = st.columns(3)
        
        with c_large:
            st.markdown("### 🥇 Mistral Large")
            res_l = comp_data["mistral-large-latest"]["run"]
            scores_l = comp_data["mistral-large-latest"]["scores"]
            st.markdown(f"**Latency:** {res_l['latency_ms']:.0f}ms")
            st.markdown(f"**Cost:** ${res_l['cost']:.6f}")
            st.markdown(f"**Response:**\n\n{res_l['response']}")
            
        with c_med:
            st.markdown("### 🥈 Mistral Medium")
            res_m = comp_data["mistral-medium-latest"]["run"]
            scores_m = comp_data["mistral-medium-latest"]["scores"]
            st.markdown(f"**Latency:** {res_m['latency_ms']:.0f}ms")
            st.markdown(f"**Cost:** ${res_m['cost']:.6f}")
            st.markdown(f"**Response:**\n\n{res_m['response']}")
            
        with c_small:
            st.markdown("### 🥉 Mistral Small")
            res_s = comp_data["mistral-small-latest"]["run"]
            scores_s = comp_data["mistral-small-latest"]["scores"]
            st.markdown(f"**Latency:** {res_s['latency_ms']:.0f}ms")
            st.markdown(f"**Cost:** ${res_s['cost']:.6f}")
            st.markdown(f"**Response:**\n\n{res_s['response']}")
            
        # Draw Plotly charts comparing metrics
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        # Build comparison DataFrame
        chart_rows = []
        for m_name, info in comp_data.items():
            run = info["run"]
            scores = info["scores"]
            chart_rows.append({
                "Model": m_name,
                "Latency (ms)": run["latency_ms"],
                "Cost ($)": run["cost"],
                "Faithfulness": scores.get("faithfulness") or 0.0,
                "Answer Relevancy": scores.get("answer_relevancy") or 0.0,
                "Context Precision": scores.get("context_precision") or 0.0,
                "Context Recall": scores.get("context_recall") or 0.0
            })
        df_comp = pd.DataFrame(chart_rows)
        
        with chart_col1:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                x=df_comp["Model"],
                y=df_comp["Latency (ms)"],
                name="Latency (ms)",
                yaxis='y1',
                marker_color='#3b82f6'
            ))
            fig_perf.add_trace(go.Scatter(
                x=df_comp["Model"],
                y=df_comp["Cost ($)"],
                name="Cost ($)",
                yaxis='y2',
                marker=dict(color='#ef4444', size=10),
                line=dict(width=2)
            ))
            fig_perf.update_layout(
                title="Performance & Cost Comparison",
                yaxis=dict(title="Latency (ms)", side="left"),
                yaxis2=dict(title="Cost ($)", side="right", overlaying="y", showgrid=False),
                template="plotly_dark",
                height=350,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with chart_col2:
            # Grouped bar chart for quality metrics
            df_melted = df_comp.melt(
                id_vars=["Model"],
                value_vars=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                var_name="Metric",
                value_name="Score"
            )
            fig_metrics = px.bar(
                df_melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="RAGAS Evaluation Scores by Model",
                range_y=[0, 1.0],
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_metrics.update_layout(height=350)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 3: EMBEDDING COMPARISON -----------------
with tab_embed_comp:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("🔍 Retrieval Quality and Embedding Benchmarking")
    st.markdown("Compare retrieval quality, precision, and search speed across Mistral Embeddings, BGE, and Sentence Transformers.")
    
    embed_query = st.text_input("Enter search query for embedding comparison:")
    embed_ground_truth = st.text_input("Expected Ground Truth text (for retrieval precision/recall comparison):")
    
    if st.button("Run Embedding Comparison", type="primary") and embed_query:
        if not api_key_valid:
            st.error("Cannot execute embedding comparison. Mistral API Key is missing.")
        else:
            embeddings_to_test = ["mistral-embed", "bge-small-en-v1.5", "sentence-transformers-all-MiniLM-L6-v2"]
            embed_results = {}
            
            with st.spinner("Retrieving source chunks from Chroma collections..."):
                progress_bar = st.progress(0)
                for i, emb_name in enumerate(embeddings_to_test):
                    # Measure retrieval latency
                    ret_start = time.time()
                    retrieved_docs = st.session_state.vectorstore_manager.retrieve_contexts(
                        query=embed_query,
                        embedding_name=emb_name,
                        top_k=top_k
                    )
                    latency_ms = (time.time() - ret_start) * 1000
                    
                    contexts = [doc.page_content for doc in retrieved_docs]
                    
                    # Run Ragas evaluators to assess Context Precision
                    # Wait, context precision compares the retrieved context to the query.
                    # We can use RagasEvaluator to compute context precision.
                    evaluator = RagasEvaluator()
                    scores = evaluator.evaluate_query(
                        query=embed_query,
                        response="Dummy response to satisfy interface", # Not evaluated for context metrics
                        contexts=contexts,
                        ground_truth=embed_ground_truth if embed_ground_truth else None
                    )
                    
                    # Save a query log representation to DB to keep histories consistent
                    # (we set response to empty as it is a retrieval evaluation)
                    db_log = QueryLogDTO(
                        query=embed_query,
                        response=f"[Retrieval comparison for {emb_name}]",
                        contexts=contexts,
                        model_name="N/A (Retrieval Test)",
                        embedding_name=emb_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        latency_ms=latency_ms,
                        session_id="retrieval_comparison"
                    )
                    log_id = st.session_state.db.save_query_log(db_log)
                    
                    eval_dto = EvaluationScoreDTO(
                        query_id=log_id,
                        context_precision=scores["context_precision"],
                        context_recall=scores["context_recall"]
                    )
                    st.session_state.db.save_evaluation(eval_dto)
                    
                    embed_results[emb_name] = {
                        "contexts": contexts,
                        "latency_ms": latency_ms,
                        "precision": scores["context_precision"],
                        "recall": scores["context_recall"]
                    }
                    progress_bar.progress((i + 1) / len(embeddings_to_test))
                    
                st.session_state.embed_results = embed_results
                st.success("Retrieval analysis completed!")
                
    if "embed_results" in st.session_state:
        emb_data = st.session_state.embed_results
        
        # Display side-by-side retrieved chunks
        e_mistral, e_bge, e_sbert = st.columns(3)
        
        with e_mistral:
            st.markdown("### 🟢 Mistral Embeddings")
            info_m = emb_data["mistral-embed"]
            st.markdown(f"**Retrieval Latency:** {info_m['latency_ms']:.1f}ms")
            st.markdown(f"**Context Precision:** {render_score_badge(info_m['precision'])}")
            st.markdown("**Top Retrieved Chunks:**")
            for idx, chunk in enumerate(info_m["contexts"][:2]):
                st.markdown(f"**Chunk {idx+1}:**")
                st.caption(chunk[:300] + "...")
                
        with e_bge:
            st.markdown("### 🔵 BGE Embeddings")
            info_b = emb_data["bge-small-en-v1.5"]
            st.markdown(f"**Retrieval Latency:** {info_b['latency_ms']:.1f}ms")
            st.markdown(f"**Context Precision:** {render_score_badge(info_b['precision'])}")
            st.markdown("**Top Retrieved Chunks:**")
            for idx, chunk in enumerate(info_b["contexts"][:2]):
                st.markdown(f"**Chunk {idx+1}:**")
                st.caption(chunk[:300] + "...")
                
        with e_sbert:
            st.markdown("### 🟣 Sentence Transformers")
            info_s = emb_data["sentence-transformers-all-MiniLM-L6-v2"]
            st.markdown(f"**Retrieval Latency:** {info_s['latency_ms']:.1f}ms")
            st.markdown(f"**Context Precision:** {render_score_badge(info_s['precision'])}")
            st.markdown("**Top Retrieved Chunks:**")
            for idx, chunk in enumerate(info_s["contexts"][:2]):
                st.markdown(f"**Chunk {idx+1}:**")
                st.caption(chunk[:300] + "...")
                
        # Draw Plotly Charts comparing embeddings
        st.markdown("---")
        ec1, ec2 = st.columns(2)
        
        embed_rows = []
        for name, info in emb_data.items():
            embed_rows.append({
                "Embedding Model": name,
                "Latency (ms)": info["latency_ms"],
                "Context Precision": info["precision"] or 0.0,
                "Context Recall": info["recall"] or 0.0
            })
        df_emb = pd.DataFrame(embed_rows)
        
        with ec1:
            fig_el = px.bar(
                df_emb,
                x="Embedding Model",
                y="Latency (ms)",
                title="Retrieval Latency Comparison",
                color="Embedding Model",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_el.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_el, use_container_width=True)
            
        with ec2:
            df_emb_melted = df_emb.melt(
                id_vars=["Embedding Model"],
                value_vars=["Context Precision", "Context Recall"],
                var_name="Metric",
                value_name="Score"
            )
            fig_em = px.bar(
                df_emb_melted,
                x="Embedding Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Retrieval Precision & Recall",
                range_y=[0.0, 1.0],
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_em.update_layout(height=300)
            st.plotly_chart(fig_em, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 4: QUERY LOGS -----------------
with tab_logs:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("📝 Observability Ledger & Query Logs")
    st.markdown("Browse, search, and audit past model logs and corresponding RAGAS metrics.")
    
    # Read joined logs from SQLite
    logs = st.session_state.db.get_logs_with_evaluations()
    
    if not logs:
        st.info("No query logs available yet. Ask a question in the first tab to populate the database.")
    else:
        df_logs = pd.DataFrame(logs)
        
        # Filter filters
        model_options = ["All"] + list(df_logs["model_name"].unique())
        embedding_options = ["All"] + list(df_logs["embedding_name"].unique())
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            search_query = st.text_input("🔍 Search Query/Response:")
        with f_col2:
            model_filter = st.selectbox("LLM Model Filter", model_options)
        with f_col3:
            embedding_filter = st.selectbox("Embedding Model Filter", embedding_options)
            
        # Apply filters
        df_filtered = df_logs.copy()
        if search_query:
            df_filtered = df_filtered[
                df_filtered["query"].str.contains(search_query, case=False) |
                df_filtered["response"].str.contains(search_query, case=False)
            ]
        if model_filter != "All":
            df_filtered = df_filtered[df_filtered["model_name"] == model_filter]
        if embedding_filter != "All":
            df_filtered = df_filtered[df_filtered["embedding_name"] == embedding_filter]
            
        st.markdown(f"Showing **{len(df_filtered)}** logs of {len(df_logs)} total:")
        
        # Render table with a details selection
        selected_log_id = st.selectbox(
            "Select a Query Log ID to inspect in detail:",
            options=df_filtered["id"].tolist(),
            format_func=lambda x: f"Log {x} | {df_filtered[df_filtered['id'] == x]['query'].iloc[0][:50]}..."
        )
        
        # Display main logs table (concise view)
        table_view = df_filtered[[
            "id", "query", "model_name", "embedding_name", "latency_ms", "cost", 
            "total_tokens", "faithfulness", "answer_relevancy", "context_precision"
        ]].rename(columns={
            "id": "ID", "query": "Query", "model_name": "Model", "embedding_name": "Embedding", 
            "latency_ms": "Latency (ms)", "cost": "Cost ($)", "total_tokens": "Tokens",
            "faithfulness": "Faithfulness", "answer_relevancy": "Relevancy", "context_precision": "Precision"
        })
        st.dataframe(table_view, use_container_width=True, hide_index=True)
        
        # Display detailed expansion card for the selected ID
        if selected_log_id:
            selected_row = df_filtered[df_filtered["id"] == selected_log_id].iloc[0]
            st.markdown("---")
            st.markdown(f"### 🔍 Detailed Inspection: Log ID {selected_log_id}")
            
            det_col1, det_col2 = st.columns([2, 1])
            with det_col1:
                st.markdown(f"**Query:**\n{selected_row['query']}")
                st.markdown(f"**Response:**\n{selected_row['response']}")
                with st.expander("Retrieved Context Chunks"):
                    for i, context in enumerate(selected_row["contexts"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(context, language="markdown")
            with det_col2:
                st.markdown("#### Execution Parameters")
                st.markdown(f"- **LLM Model:** `{selected_row['model_name']}`")
                st.markdown(f"- **Embedding Model:** `{selected_row['embedding_name']}`")
                st.markdown(f"- **Chunk Setup:** Size `{selected_row['chunk_size']}` | Overlap `{selected_row['chunk_overlap']}`")
                st.markdown(f"- **Latency:** `{selected_row['latency_ms']:.1f} ms`")
                st.markdown(f"- **Cost:** `${selected_row['cost']:.6f}`")
                st.markdown(f"- **Token Usage:** `{selected_row['total_tokens']}` (Prompt: {selected_row['prompt_tokens']} | Gen: {selected_row['completion_tokens']})")
                st.markdown(f"- **Log Date:** `{selected_row['created_at']}`")
                
                st.markdown("#### Evaluation Results")
                st.markdown(f"- **Faithfulness:** {render_score_badge(selected_row.get('faithfulness'))}")
                st.markdown(f"- **Answer Relevancy:** {render_score_badge(selected_row.get('answer_relevancy'))}")
                st.markdown(f"- **Context Precision:** {render_score_badge(selected_row.get('context_precision'))}")
                st.markdown(f"- **Context Recall:** {render_score_badge(selected_row.get('context_recall'))}")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 5: SYSTEM ANALYTICS -----------------
with tab_analytics:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("📈 Overall System Analytics & Financial Observability")
    st.markdown("Aggregated analytical metrics and historical performance distributions across all logged executions.")
    
    analytics = st.session_state.db.get_analytics_summary()
    
    # 1. Row of big KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div class='card-header'>Total Queries Logged</div>
            <div class='card-value card-value-highlight'>{analytics['total_queries']}</div>
            <div class='card-subtitle'>Database: SQLite ({DB_PATH.name})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div class='card-header'>Total Financial Cost</div>
            <div class='card-value card-value-highlight'>${analytics['total_cost']:.4f}</div>
            <div class='card-subtitle'>Tokens Consumed: {analytics['total_tokens']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div class='card-header'>Average Latency</div>
            <div class='card-value card-value-highlight'>{analytics['avg_latency']:.0f}ms</div>
            <div class='card-subtitle'>Includes retrieval + LLM call</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        avg_faith = analytics.get('avg_faithfulness')
        faith_val = f"{avg_faith:.2f}" if avg_faith is not None else "N/A"
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div class='card-header'>Avg Faithfulness</div>
            <div class='card-value card-value-highlight'>{faith_val}</div>
            <div class='card-subtitle'>Evaluated with RAGAS LLM judge</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Charts section
    st.markdown("---")
    c_col1, c_col2 = st.columns(2)
    
    # Trend over time (Cost)
    with c_col1:
        if analytics["cost_trends"]:
            df_trends = pd.DataFrame(analytics["cost_trends"])
            fig_trend = px.area(
                df_trends,
                x="date_day",
                y="cost_sum",
                title="Accumulated Daily Financial Cost ($)",
                labels={"date_day": "Date", "cost_sum": "Cost ($)"},
                template="plotly_dark",
                color_discrete_sequence=['#8b5cf6']
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No trend data available yet.")
            
    # Aggregated Model distributions
    with c_col2:
        model_comp_data = st.session_state.db.get_model_comparison_data()
        if model_comp_data:
            df_model_aggr = pd.DataFrame(model_comp_data)
            fig_pie = px.pie(
                df_model_aggr,
                values="query_count",
                names="model_name",
                title="Query Distribution by LLM Model",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No query distribution data available.")

    # 3. Aggregated metrics table by model/embeddings
    st.markdown("---")
    aggr_col1, aggr_col2 = st.columns(2)
    
    with aggr_col1:
        st.markdown("#### LLM Model Performance Matrix")
        model_comp_data = st.session_state.db.get_model_comparison_data()
        if model_comp_data:
            df_m = pd.DataFrame(model_comp_data)
            # Rename columns for display
            df_m_disp = df_m.rename(columns={
                "model_name": "Model",
                "query_count": "Runs",
                "avg_latency_ms": "Avg Latency (ms)",
                "avg_cost": "Avg Cost ($)",
                "avg_faithfulness": "Faithfulness",
                "avg_answer_relevancy": "Answer Relevancy",
                "avg_context_precision": "Context Precision",
                "avg_context_recall": "Context Recall"
            })
            st.dataframe(df_m_disp, use_container_width=True, hide_index=True)
        else:
            st.info("No aggregated model matrix available yet.")
            
    with aggr_col2:
        st.markdown("#### Embedding Model Retrieval Matrix")
        embed_comp_data = st.session_state.db.get_embedding_comparison_data()
        if embed_comp_data:
            df_e = pd.DataFrame(embed_comp_data)
            df_e_disp = df_e.rename(columns={
                "embedding_name": "Embedding",
                "query_count": "Runs",
                "avg_latency_ms": "Avg Latency (ms)",
                "avg_cost": "Avg Cost ($)",
                "avg_context_precision": "Context Precision",
                "avg_context_recall": "Context Recall"
            })
            st.dataframe(df_e_disp, use_container_width=True, hide_index=True)
        else:
            st.info("No aggregated embedding matrix available yet.")
            
    st.markdown("</div>", unsafe_allow_html=True)
