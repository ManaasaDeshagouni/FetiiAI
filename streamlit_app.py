import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from datetime import datetime
from query_engine import (
    load_and_prepare_data,
    top_dropoffs_by_age_group_and_day,
    peak_hours_for_large_groups,
    trips_to_specific_location,
    age_distribution_at_location,
    group_size_by_day_of_week,
    least_busy_locations_by_day,
    busiest_hours_by_location,
    weekend_vs_weekday_patterns,
    natural_summary,
    friendly_response,
)
from enhanced_gpt_router_v3 import enhanced_gpt_route_v3

# === Page Setup ===
st.set_page_config(page_title="FetiiAI Chatbot", layout="wide", initial_sidebar_state="expanded")

TELEMETRY_PATH = "telemetry.csv"

def log_event(question: str, route: dict, status: str, rows: int | None = None, func: str | None = None):
    try:
        exists = os.path.exists(TELEMETRY_PATH)
        with open(TELEMETRY_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp","question","status","route_type","function","rows"])
            writer.writerow([
                datetime.utcnow().isoformat(),
                question,
                status,
                route.get("type") if isinstance(route, dict) else None,
                func,
                rows,
            ])
    except Exception:
        pass

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: transparent;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 1rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.8rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.5;
    }
    .main-header .subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 0.95rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Chat Container */
    .chat-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        background-size: 100% 100%;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        min-height: 86px; /* ensure same height even if text is short */
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        line-height: 1.3;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
        white-space: normal; /* allow wrapping without shrinking height */
        word-break: break-word;
    }
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        outline: none;
    }
    
    /* Quick Question Buttons */
    .quick-question-btn {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #e2e8f0;
        color: #475569;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        margin: 0.25rem;
        text-align: center;
        cursor: pointer;
    }
    .quick-question-btn:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-color: #3b82f6;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    
    /* Footer */
    .footer-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced button hover effects */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.2s ease;
    }
    
    /* Better spacing for main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhanced input focus */
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15);
        transform: scale(1.02);
    }
    
    /* Professional card hover effects */
    .card-hover:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>🚐 FetiiAI</h1>
    <p>Intelligent Rideshare Analytics Platform</p>
    <p class="subtitle">Powered by GPT-4 • Austin, Texas</p>
</div>
""", unsafe_allow_html=True)

# === Load and cache data ===
@st.cache_data
def load_data():
    return load_and_prepare_data("FetiiAI_Data_Austin.xlsx")

df = load_data()

# === Professional Sidebar ===
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h2 style='color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 600;'>Analytics Dashboard</h2>
    <p style='color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Explore Austin Rideshare Data</p>
</div>
""", unsafe_allow_html=True)

# Quick Actions Section
st.sidebar.markdown("### 🎯 Quick Actions")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📊 Overview", use_container_width=True, help="Get a comprehensive data overview"):
        st.session_state.quick_action = "data_overview"
with col2:
    if st.button("🎲 Insight", use_container_width=True, help="Discover random insights"):
        st.session_state.quick_action = "random_insight"

st.sidebar.markdown("---")

# Quick Stats Section
st.sidebar.markdown("### 📊 Quick Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Trips", f"{len(df):,}", help="Total rideshare trips in dataset")
with col2:
    st.metric("Locations", f"{df['dropoff_simple'].nunique():,}", help="Unique dropoff locations")

st.sidebar.markdown("---")

# Data Filters Section
st.sidebar.markdown("### 🔍 Data Filters")
st.sidebar.markdown("*Filter your analysis by specific criteria*")

selected_day = st.sidebar.selectbox(
    "Day of Week", 
    ["All"] + sorted(df['day_of_week'].unique().tolist()),
    help="Filter by specific day of the week"
)
selected_age = st.sidebar.selectbox(
    "Age Group", 
    ["All"] + sorted(df['age_group'].unique().tolist()),
    help="Filter by age demographics"
)

st.sidebar.markdown("---")

# Conversation History Section
st.sidebar.markdown("### 💬 Conversation History")
if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    st.sidebar.success(f"📝 {len(st.session_state.history)} exchanges")
    for i, (q, a) in enumerate(st.session_state.history[-5:]):  # Show last 5
        with st.sidebar.expander(f"Q{i+1}: {q[:25]}...", expanded=False):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a[:100]}{'...' if len(a) > 100 else ''}")
else:
    st.sidebar.info("💡 Start asking questions to build your conversation history!")

# Clear history button
if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    st.session_state.history = []
    st.rerun()

# === Initialize session state for user input ===
if 'user_input' not in st.session_state:
    st.session_state.user_input = None

# === Quick Actions Handler ===
if hasattr(st.session_state, 'quick_action'):
    if st.session_state.quick_action == "data_overview":
        st.session_state.user_input = "Show me a data overview of the rideshare patterns"
    elif st.session_state.quick_action == "random_insight":
        import random
        random_questions = [
            "What are the most popular dropoff locations?",
            "When do large groups ride most?",
            "What's the age distribution of riders?",
            "Where should I avoid on weekends?",
            "What are the busiest hours?"
        ]
        st.session_state.user_input = random.choice(random_questions)
    del st.session_state.quick_action

# === Enhanced Input Section ===
# Removed 'Ask Your Question' header card

# === User Input ===
if not st.session_state.user_input:
    user_input = st.text_input(
        "What would you like to know about Austin rideshare data?", 
        placeholder="e.g., Where do teens go on Friday nights?",
        help="Ask questions about patterns, demographics, locations, or trends"
    )
    if user_input:
        st.session_state.user_input = user_input
        st.rerun()
else:
    user_input = st.session_state.user_input

# === Enhanced Quick Questions ===
# Removed 'Popular Questions' header card

# Create a more organized grid layout
col1, col2, col3 = st.columns(3)
suggestions = [
    ("Where do teens go on Friday?", "👥", "Teen demographics"),
    ("When do big groups ride?", "🚌", "Group patterns"), 
    ("What's the quietest day?", "😴", "Low activity"),
    ("Show me weekend patterns", "📅", "Weekend trends"),
    ("Where should I avoid crowds?", "🚫", "Crowd avoidance"),
    ("What ages use Fetii most?", "📊", "Age analysis")
]

for i, (suggestion, icon, category) in enumerate(suggestions):
    with [col1, col2, col3][i % 3]:
        st.markdown(f"""
        <div style='background: transparent; padding: 0.25rem 0; margin: 0.25rem 0; text-align: center;'>
            <div style='font-size: 1.5rem; margin-bottom: 0.35rem;'>{icon}</div>
            <p style='color: #94a3b8; margin: 0; font-size: 0.8rem; font-weight: 500;'>{category}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(suggestion, use_container_width=True, key=f"suggestion_{i}"):
            st.session_state.user_input = suggestion
            st.rerun()

# === Process User Input and Show Answers ===
if user_input:
    # Initialize placeholders
    response_text = ""
    chart = None
    df_out = None
    thought_text = ""
    natural_summary_text = ""
    suggestions = []
    show_data_table = False

    # 🔄 Route input through Enhanced GPT Router 3.0 (Function Calling) with conversation context
    conversation_history = st.session_state.get('history', [])
    route = enhanced_gpt_route_v3(user_input, conversation_history)

    # 🧠 Debug expander (hidden by default, only for development)
    if st.session_state.get('debug_mode', False):
        with st.expander("🛠️ Debug: GPT Routing Output"):
            st.json(route)

    if route:
        func_type = route.get("type")
        func = route.get("function")
        params = route.get("parameters", {})
        thought_text = route.get("thought", "")
        suggestions = route.get("suggestions", [])
        pre_response = route.get("response", "")

        try:
            if func_type == "function_call":
                # Perform actual analytics call
                if func == "top_dropoffs_by_age_group_and_day":
                    df_out, chart = top_dropoffs_by_age_group_and_day(df, **params)
                elif func == "peak_hours_for_large_groups":
                    df_out, chart = peak_hours_for_large_groups(df, **params)
                elif func == "trips_to_specific_location":
                    df_out, chart = trips_to_specific_location(df, **params)
                elif func == "age_distribution_at_location":
                    df_out, chart = age_distribution_at_location(df, **params)
                elif func == "group_size_by_day_of_week":
                    df_out, chart = group_size_by_day_of_week(df)
                elif func == "least_busy_locations_by_day":
                    df_out, chart = least_busy_locations_by_day(df, **params)
                elif func == "busiest_hours_by_location":
                    df_out, chart = busiest_hours_by_location(df, **params)
                elif func == "weekend_vs_weekday_patterns":
                    df_out, chart = weekend_vs_weekday_patterns(df)

                # Generate dynamic summary using real df_out
                if df_out is not None and len(df_out) > 0:
                    summary = natural_summary(df_out, context=func.replace("_", " "))
                    response_text = f"**{summary}**"
                    # Only show small tables inline
                    if len(df_out) < 20:
                        response_text += "\n\n" + df_out.to_html(index=False)
                else:
                    response_text = pre_response or "No results found for that query. Try changing the day, age group, or location."
            elif func_type == "natural_response":
                response_text = pre_response or "I couldn't route that to a specific analysis, but I'm here to help!"
        except Exception as e:
            response_text = f"❌ Error occurred: {str(e)}"

        # Attach suggestions if any
        if suggestions:
            response_text += "\n\n**💡 Try asking:**\n" + "\n".join(f"• {s}" for s in suggestions)

        # Save to history
        st.session_state.history.append((user_input, response_text))
        # Telemetry
        try:
            log_event(user_input, route, status="ok" if "❌" not in response_text else "error", rows=(len(df_out) if df_out is not None else None), func=func)
        except Exception:
            pass

        # === Professional Answer Display ===
        # Only show content when there's actually something to display
        if response_text or df_out is not None or chart:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Professional answer header
            st.markdown("""
            <div style='background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
                <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>🎯 Analysis Results</h2>
                <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.95rem;'>Based on your question: "{}</p>
            </div>
            """.format(user_input), unsafe_allow_html=True)
            
            # Main answer content
            st.markdown(response_text, unsafe_allow_html=True)
            
            # Show conversation context indicator (subtle)
            if conversation_history:
                st.markdown(f"""
                <div style='background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 0.75rem; margin: 1rem 0;'>
                    <p style='color: #0369a1; margin: 0; font-size: 0.9rem;'>💬 Using context from {len(conversation_history)} previous exchanges</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Thought process (collapsed by default, professional styling)
            if thought_text:
                with st.expander("🔍 Analysis Methodology", expanded=False):
                    st.markdown(f"""
                    <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                        <p style='color: #475569; margin: 0; font-size: 0.9rem; line-height: 1.6;'>{thought_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Professional data display sections
            if df_out is not None or chart:
                st.markdown("---")
                
            # Interactive data display with professional styling
            if df_out is not None:
                st.markdown("""
                <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid #e2e8f0;'>
                    <h3 style='color: #1e293b; margin: 0 0 1rem 0; font-size: 1.2rem; font-weight: 600;'>📊 Data Insights</h3>
                    <p style='color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;'>Detailed analysis results from your query</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(df_out, use_container_width=True)
                
                # Professional download section
                col1, col2 = st.columns([3, 1])
                with col2:
                    csv = df_out.to_csv(index=False)
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv,
                        file_name=f"fetii_analysis_{func if func else 'results'}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Chart with professional styling
            if chart:
                st.markdown("""
                <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid #e2e8f0;'>
                    <h3 style='color: #1e293b; margin: 0 0 1rem 0; font-size: 1.2rem; font-weight: 600;'>📈 Visual Analysis</h3>
                    <p style='color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;'>Interactive chart showing key patterns and trends</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.pyplot(chart)
                
                # Professional chart download
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("📥 Export Chart", use_container_width=True):
                        chart.savefig("fetii_chart.png", dpi=300, bbox_inches='tight')
                        with open("fetii_chart.png", "rb") as file:
                            st.download_button(
                                label="Download PNG",
                                data=file.read(),
                                file_name="fetii_chart.png",
                                mime="image/png",
                                use_container_width=True
                            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a button to ask a new question
    if st.button("🔄 Ask New Question", use_container_width=True):
        st.session_state.user_input = None
        st.rerun()

# === Compact Additional Tools ===
st.markdown("""
<div style='background: #f8fafc; padding: 1.5rem; border-radius: 16px; margin: 2rem 0; border: 1px solid #e2e8f0;'>
    <div style='text-align: center; margin-bottom: 1.5rem;'>
        <h3 style='color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 600;'>🛠️ Additional Tools</h3>
        <p style='color: #64748b; margin: 0; font-size: 0.9rem;'>Quick actions and developer features</p>
    </div>
</div>
""", unsafe_allow_html=True)

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 12px; border: 1px solid #f59e0b; text-align: center; margin-bottom: 0.5rem;'>
        <h4 style='color: #92400e; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;'>🎯 Quick Actions</h4>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Random Insight", use_container_width=True, help="Get a random data insight"):
        import random
        random_questions = [
            "What are the most popular dropoff locations?",
            "When do large groups ride most?",
            "What's the age distribution of riders?",
            "Where should I avoid on weekends?",
            "What are the busiest hours?"
        ]
        st.session_state.random_question = random.choice(random_questions)
        st.rerun()

with footer_col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1rem; border-radius: 12px; border: 1px solid #3b82f6; text-align: center; margin-bottom: 0.5rem;'>
        <h4 style='color: #1e40af; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;'>📊 Data Explorer</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 0.75rem; text-align: center;'>
        <p style='color: #0369a1; margin: 0; font-size: 0.9rem; font-weight: 500;'>💡 Ask questions to explore the data!</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); padding: 1rem; border-radius: 12px; border: 1px solid #8b5cf6; text-align: center; margin-bottom: 0.5rem;'>
        <h4 style='color: #6b21a8; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;'>🛠️ Developer Tools</h4>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔧 Debug Mode", use_container_width=True, help="Toggle debug information"):
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = not st.session_state.debug_mode
        st.rerun()

# Handle random question
if hasattr(st.session_state, 'random_question'):
    st.info(f"🎲 Random question: {st.session_state.random_question}")
    if st.button("Ask This Question"):
        user_input = st.session_state.random_question
        del st.session_state.random_question
        st.rerun()

# Dataset summary removed - users don't care about stats

# Elegant Footer Branding
st.markdown("""
<div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 1.5rem; border-radius: 16px; margin-top: 2rem; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
    <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
        <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 12px; margin-right: 1rem;'>
            <span style='font-size: 1.5rem;'>🚐</span>
        </div>
        <div>
            <h3 style='color: white; margin: 0; font-size: 1.4rem; font-weight: 700;'>FetiiAI</h3>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem; font-weight: 500;'>Intelligent Analytics Platform</p>
        </div>
    </div>
    <div style='border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;'>
        <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;'>Powered by GPT-4 • Built with Streamlit • Austin, Texas</p>
        <p style='color: rgba(255,255,255,0.6); margin: 0.5rem 0 0 0; font-size: 0.75rem;'>© 2024 FetiiAI - Advanced Data Analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)