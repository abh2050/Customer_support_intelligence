import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Initialize the Gemini client
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Set page config
st.set_page_config(
    page_title="Customer Support Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data(use_sample_data=False):
    """
    Load data for the Streamlit app.
    
    Args:
        use_sample_data (bool): If True, use sample data instead of full data
    """
    # For Streamlit Cloud deployment, always use sample data
    # Check for deployment environment or explicitly requested sample data
    if use_sample_data or os.environ.get('STREAMLIT_SHARING') == 'true' or not os.path.exists('clustering_results.pkl'):
        st.info("Using generated sample data for demonstration purposes.")
        
        # Generated sample data with consistent structure
        num_samples = 100
        
        # Create synthetic embeddings data
        embedding_data = {
            'embeddings': [np.random.rand(768).astype(np.float32) for _ in range(num_samples)],
            'texts': [f"Sample customer issue #{i}: Having trouble with my account" for i in range(num_samples)],
            'indices': list(range(num_samples))
        }
        
        # Create synthetic cluster data
        cluster_data = {
            'embeddings_2d': np.random.rand(num_samples, 2).astype(np.float32),
            'clusters': np.random.randint(0, 6, num_samples),
            'texts': embedding_data['texts'],
            'indices': embedding_data['indices'],
            'optimal_clusters': 6
        }
        
    else:
        # Only try to load real data if we're in a local development environment
        # and have checked that the files exist
        try:
            # Load clustering results only (avoid the larger embeddings file)
            with open('clustering_results.pkl', 'rb') as f:
                cluster_data = pickle.load(f)
            
            # Extract embeddings from cluster data if possible
            if 'texts' in cluster_data:
                # Generate synthetic embeddings that match the texts from cluster data
                # This avoids needing the large combined_embeddings.pkl file
                texts = cluster_data['texts']
                indices = cluster_data['indices']
                embedding_data = {
                    'embeddings': [np.random.rand(768).astype(np.float32) for _ in range(len(texts))],
                    'texts': texts,
                    'indices': indices
                }
                st.success("Loaded data with synthesized embeddings.")
            else:
                # Fall back to sample data if cluster data doesn't have necessary fields
                st.warning("Incomplete data in clustering_results.pkl. Using sample data instead.")
                return load_data(use_sample_data=True)
                
        except Exception as e:
            st.warning(f"Error loading data files: {e}. Using sample data instead.")
            return load_data(use_sample_data=True)
    
    # Create a basic cluster information dictionary (same for both real and sample data)
    clusters_info = {
        0: {"name": "Account Access Issues", 
            "description": "Problems with logging in, account verification, and security"},
        1: {"name": "Billing Problems", 
            "description": "Issues related to charges, refunds, and payment processing"},
        2: {"name": "Product Quality Concerns", 
            "description": "Complaints about product functionality, durability, or expectations"},
        3: {"name": "Delivery Delays", 
            "description": "Issues with shipping, tracking, and delivery timeframes"},
        4: {"name": "Website Technical Issues", 
            "description": "Problems with website functionality, errors, and user experience"},
        5: {"name": "Customer Service Experience", 
            "description": "Feedback about interactions with support staff and resolution processes"},
    }
    
    return cluster_data, embedding_data, clusters_info

# Function to classify new text with Gemini
def classify_with_gemini(text, cluster_info):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create a prompt with cluster descriptions
        prompt = "You are a customer support issue classifier. Based on the cluster descriptions below, determine which cluster this customer support issue belongs to:\n\n"
        
        for cluster_id, info in cluster_info.items():
            prompt += f"Cluster {cluster_id}: {info['name']} - {info['description']}\n"
        
        prompt += f"\nCustomer Issue: {text}\n\nRespond with the cluster number only (0-5)."
        
        response = model.generate_content(prompt)
        
        # Extract just the cluster number
        cluster_text = response.text.strip()
        try:
            # Try to extract a number from the response
            for word in cluster_text.split():
                if word.isdigit() and 0 <= int(word) <= 5:
                    return int(word)
            
            # If specific patterns failed, look for any digit
            import re
            matches = re.findall(r'\d', cluster_text)
            if matches and 0 <= int(matches[0]) <= 5:
                return int(matches[0])
                
            # Default fallback
            return 0
        except:
            return 0
    except Exception as e:
        st.error(f"Error classifying with Gemini: {str(e)}")
        return 0

# Function to get solution recommendation using Gemini
def get_solution_recommendation(text, cluster_id, cluster_info):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        cluster_name = cluster_info[cluster_id]["name"]
        cluster_description = cluster_info[cluster_id]["description"]
        
        prompt = f"""
        As an AI customer support assistant, recommend a solution for the following customer issue:
        
        Customer Issue: {text}
        
        This issue has been classified as: {cluster_name} ({cluster_description})
        
        Provide a helpful, empathetic response that includes:
        1. Acknowledgment of the specific issue
        2. A step-by-step solution
        3. Any preventative measures to avoid this issue in the future
        4. A clear next step if the recommendation doesn't resolve the issue
        
        Format your response in a way that a customer service agent could use it directly.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting recommendation: {str(e)}")
        return "Unable to generate recommendation. Please try again later."

# Function to analyze issue trends with Gemini
def analyze_trends_with_gemini(cluster_trends):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert trend data to a simple text representation
        trend_text = "Issue cluster trends over the past weeks:\n\n"
        for cluster_id, trend in cluster_trends.items():
            trend_text += f"Cluster {cluster_id}: {trend}\n"
        
        prompt = f"""
        As a customer support analytics expert, analyze the following trends in customer issues:
        
        {trend_text}
        
        Please provide:
        1. Key insights about emerging or declining issues
        2. Any notable patterns or correlations between different issue types
        3. Actionable recommendations for the support team based on these trends
        
        Format your response as a concise executive summary with bullet points.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing trends: {str(e)}")
        return "Unable to analyze trends. Please try again later."

# Function to embed text for similarity search
def embed_text(text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

# Main function
def main():
    # Load data
    try:
        # Try loading real data first, fall back to sample data if needed
        cluster_data, embedding_data, clusters_info = load_data()
        embeddings_array = np.array(embedding_data['embeddings'])
        texts = embedding_data['texts']
        clusters = cluster_data['clusters']
    except Exception as e:
        st.error(f"Error setting up data: {str(e)}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Customer Support Intelligence")
    
    # Navigation
    page = st.sidebar.radio("Navigation", 
                           ["Dashboard", "Issue Classifier", "Similar Issues Finder", 
                            "Solution Recommender", "Trend Analysis"])
    
    # Add color coding to clusters
    cluster_colors = {
        0: "#FF9999",  # red
        1: "#66B2FF",  # blue
        2: "#99FF99",  # green
        3: "#FFCC99",  # orange
        4: "#CC99FF",  # purple
        5: "#FFFF99",  # yellow
    }
    
    # Dashboard
    if page == "Dashboard":
        st.title("Customer Support Intelligence Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Issue Distribution by Cluster")
            
            # Count issues per cluster
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            # Create a DataFrame for plotting
            cluster_df = pd.DataFrame({
                'Cluster': [f"{clusters_info[i]['name']} (Cluster {i})" for i in cluster_counts.index],
                'Count': cluster_counts.values
            })
            
            # Create plot
            fig = px.bar(cluster_df, x='Cluster', y='Count', 
                         color='Cluster', 
                         color_discrete_sequence=px.colors.qualitative.Bold)
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Issue Evolution Over Time")
            
            # Sample evolution data (would be real data in production)
            dates = pd.date_range(start='2017-10-01', end='2017-12-01', freq='W')
            evolution_data = {}
            
            for i in range(6):  # 6 clusters
                # Generate some random trend data
                if i in cluster_counts.index:
                    baseline = cluster_counts[i] / 10
                    evolution_data[i] = [max(0, baseline + np.random.randint(-5, 10)) for _ in range(len(dates))]
                else:
                    evolution_data[i] = [0] * len(dates)
            
            # Create DataFrame
            evolution_df = pd.DataFrame(evolution_data, index=dates)
            evolution_df.columns = [f"Cluster {i}" for i in evolution_df.columns]
            
            # Create plot with plotly
            fig = go.Figure()
            
            for column in evolution_df.columns:
                cluster_num = int(column.split(' ')[1])
                fig.add_trace(go.Scatter(
                    x=evolution_df.index, 
                    y=evolution_df[column],
                    mode='lines+markers',
                    name=f"{clusters_info[cluster_num]['name']} ({column})",
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Issues",
                legend=dict(orientation="h", y=-0.2)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample issues by cluster
        st.subheader("Sample Issues by Cluster")
        
        # Create tabs for each cluster
        tabs = st.tabs([f"Cluster {i}: {clusters_info[i]['name']}" for i in range(6)])
        
        for i, tab in enumerate(tabs):
            with tab:
                cluster_mask = np.where(clusters == i)[0]
                if len(cluster_mask) > 0:
                    sample_indices = np.random.choice(cluster_mask, min(5, len(cluster_mask)), replace=False)
                    sample_texts = [texts[idx] for idx in sample_indices]
                    
                    for j, text in enumerate(sample_texts):
                        st.markdown(f"**Issue {j+1}:** {text[:150]}..." if len(text) > 150 else f"**Issue {j+1}:** {text}")
                        st.markdown("---")
                else:
                    st.write("No issues in this cluster.")
    
    # Issue Classifier
    elif page == "Issue Classifier":
        st.title("Customer Issue Classifier")
        st.write("Enter a customer support issue to classify it into one of our issue categories.")
        
        # Input box for customer issue
        issue_text = st.text_area("Customer Support Issue:", height=150,
                                  placeholder="Type or paste a customer issue here...")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Classify Issue", type="primary"):
                if issue_text:
                    with st.spinner("Classifying issue..."):
                        # Get cluster prediction
                        cluster_id = classify_with_gemini(issue_text, clusters_info)
                        
                        # Display result
                        st.success(f"Issue classified as: Cluster {cluster_id}")
                        
                        # Highlight the cluster name and description
                        st.markdown(f"""
                        ### {clusters_info[cluster_id]['name']}
                        {clusters_info[cluster_id]['description']}
                        """)
                else:
                    st.error("Please enter an issue to classify.")
        
        with col2:
            # Show cluster information
            st.subheader("Issue Categories")
            for cluster_id, info in clusters_info.items():
                with st.expander(f"Cluster {cluster_id}: {info['name']}"):
                    st.write(info['description'])
                    
                    # Get sample texts for this cluster
                    cluster_mask = np.where(clusters == cluster_id)[0]
                    if len(cluster_mask) > 0:
                        sample_idx = np.random.choice(cluster_mask, 1)[0]
                        st.markdown(f"**Example:** {texts[sample_idx]}")
    
    # Similar Issues Finder
    elif page == "Similar Issues Finder":
        st.title("Find Similar Customer Issues")
        st.write("Enter a customer issue to find similar past issues and how they were resolved.")
        
        # Input box for customer issue
        issue_text = st.text_area("Customer Support Issue:", height=150,
                                  placeholder="Type or paste a customer issue here...")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            num_results = st.slider("Number of similar issues to find:", 1, 10, 5)
            
            if st.button("Find Similar Issues", type="primary"):
                if issue_text:
                    with st.spinner("Finding similar issues..."):
                        # Get embedding for the input text
                        query_embedding = embed_text(issue_text)
                        
                        if query_embedding:
                            # Calculate similarity
                            similarities = cosine_similarity([query_embedding], embeddings_array)[0]
                            
                            # Get top N similar issues
                            top_indices = np.argsort(similarities)[-num_results:][::-1]
                            top_scores = similarities[top_indices]
                            
                            # Get cluster prediction for the input text
                            cluster_id = classify_with_gemini(issue_text, clusters_info)
                            
                            # Display result
                            st.success(f"Issue classified as: {clusters_info[cluster_id]['name']} (Cluster {cluster_id})")
                            
                            # Display similar issues
                            st.subheader("Similar Past Issues:")
                            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                                with st.container():
                                    st.markdown(f"### {i+1}. Similar Issue (Similarity: {score:.2f})")
                                    st.markdown(f"**Text:** {texts[idx]}")
                                    text_cluster = clusters[idx]
                                    st.markdown(f"**Category:** {clusters_info[text_cluster]['name']} (Cluster {text_cluster})")
                                    st.markdown("---")
                else:
                    st.error("Please enter an issue to find similar cases.")
    
    # Solution Recommender
    elif page == "Solution Recommender":
        st.title("Solution Recommender")
        st.write("Get AI-powered solution recommendations for customer issues.")
        
        # Input box for customer issue
        issue_text = st.text_area("Customer Support Issue:", height=150,
                                  placeholder="Type or paste a customer issue here...")
        
        if st.button("Generate Recommendation", type="primary"):
            if issue_text:
                with st.spinner("Analyzing issue and generating recommendation..."):
                    # Get cluster prediction
                    cluster_id = classify_with_gemini(issue_text, clusters_info)
                    
                    # Get solution recommendation
                    recommendation = get_solution_recommendation(issue_text, cluster_id, clusters_info)
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.success(f"Issue classified as: {clusters_info[cluster_id]['name']} (Cluster {cluster_id})")
                        
                        # Create a priority indicator
                        priority = "High" if any(word in issue_text.lower() for word in ["urgent", "immediately", "asap", "emergency"]) else "Standard"
                        
                        priority_color = "#FF0000" if priority == "High" else "#00CC00"
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: {priority_color}; color: white; text-align: center; font-weight: bold;">
                        {priority} Priority
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Recommended Solution:")
                        st.markdown(recommendation)
            else:
                st.error("Please enter an issue to get a recommendation.")
    
    # Trend Analysis
    elif page == "Trend Analysis":
        st.title("Customer Issue Trend Analysis")
        st.write("Analyze trends in customer issues over time and get AI-powered insights.")
        
        # Sample trend data
        dates = pd.date_range(start='2017-10-01', end='2017-12-01', freq='W')
        trend_data = {}
        
        for i in range(6):  # 6 clusters
            # Generate some trend data
            if i == 0:  # Account issues - increasing trend
                trend_data[i] = [20 + j*2 + np.random.randint(-3, 4) for j in range(len(dates))]
            elif i == 1:  # Billing issues - stable with spike
                trend_data[i] = [30 + np.random.randint(-3, 4) for _ in range(len(dates))]
                trend_data[i][5] = 45  # Add a spike
            elif i == 2:  # Product quality - decreasing
                trend_data[i] = [40 - j + np.random.randint(-3, 4) for j in range(len(dates))]
            else:
                trend_data[i] = [15 + np.random.randint(-5, 6) for _ in range(len(dates))]
        
        # Create DataFrame
        trend_df = pd.DataFrame(trend_data, index=dates)
        trend_df.columns = [f"Cluster {i}" for i in trend_df.columns]
        
        # Plot trends
        st.subheader("Issue Trends by Category")
        
        # Create plot with plotly
        fig = go.Figure()
        
        for column in trend_df.columns:
            cluster_num = int(column.split(' ')[1])
            fig.add_trace(go.Scatter(
                x=trend_df.index, 
                y=trend_df[column],
                mode='lines+markers',
                name=f"{clusters_info[cluster_num]['name']} ({column})",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Number of Issues",
            legend=dict(orientation="h", y=-0.2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-powered trend analysis
        st.subheader("AI-Powered Trend Insights")
        
        if st.button("Generate Trend Analysis", type="primary"):
            with st.spinner("Analyzing trends..."):
                # Create trend descriptions
                trend_descriptions = {
                    0: "Steadily increasing over the past weeks (+10%)",
                    1: "Stable with a significant spike in week 5 (+50%)",
                    2: "Gradually decreasing over time (-15%)",
                    3: "Fluctuating with no clear trend",
                    4: "Slight increase followed by stabilization",
                    5: "Low volume with occasional small spikes"
                }
                
                # Get AI analysis
                analysis = analyze_trends_with_gemini(trend_descriptions)
                
                # Display analysis
                st.markdown(analysis)

if __name__ == "__main__":
    main()
