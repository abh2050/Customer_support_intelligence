# Customer Support Intelligence EDA System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-support-intelligence.streamlit.app/)

## Overview

A comprehensive customer support intelligence system that leverages NLP embeddings and Gemini AI to analyze, classify, and provide insights on customer support issues.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Customer+Support+Dashboard)

## Key Features

### Data Analysis & Insights
- **Root Cause Analysis**: Identify underlying causes of customer issues through semantic clustering
- **Trend Tracking**: Monitor how problem patterns evolve over time
- **Solution Recommendation**: Suggest solutions based on historical resolution data
- **Issue Clustering**: Group semantically similar customer problems

### Technical Implementation
- **Batch Embedding Generation**: Process large datasets efficiently with rate limiter protection
- **Gemini AI Integration**: Enhanced analysis and recommendations using Google's Gemini API
- **Interactive Streamlit UI**: User-friendly interface for exploring and utilizing insights

## Dataset

This project utilizes the [Kaggle Customer Support Dataset](https://lnkd.in/dCgdzs6D), which contains over 3 million customer support tweets from major brands.

## System Architecture

1. **Data Processing Pipeline**:
   - Text cleaning and normalization
   - Batch embedding generation with rate limiting
   - Semantic clustering of support issues

2. **Analysis Engine**:
   - Topic identification across customer issues
   - Time-series analysis of issue patterns
   - Escalation prediction modeling

3. **User Interface**:
   - Interactive dashboard for issue overview
   - Issue classifier for new support tickets
   - Solution recommender for support agents
   - Trend analysis with AI-powered insights

## Setup Instructions

### Prerequisites
- Python 3.9+
- Gemini API key

### Installation

1. Clone the repository:
```
2. Install dependencies:
```
3. Create a `.env` file with your Gemini API key:
```
4. Run the Streamlit app:
```

## Usage

### Dashboard
View overall distribution of customer issues and track trends over time.

### Issue Classifier
Enter a new customer issue to classify it into the appropriate category.

### Similar Issues Finder
Find semantically similar past issues to reference previous resolutions.

### Solution Recommender
Get AI-powered solution recommendations for specific customer problems.

### Trend Analysis
Analyze how customer issues evolve over time with AI-generated insights.

## Technical Details

### Embedding Process
The system uses a batch processing approach with rate limiting to generate embeddings from the Gemini API:

- **Batch Size**: 40 items per request
- **Rate Limiting**: Respects API constraints (up to 1200 RPM)
- **Persistence**: Saves embeddings to allow incremental processing
- **Error Handling**: Robust error management and progress tracking

### Clustering Methodology
Customer issues are clustered using:

- **Dimensionality Reduction**: UMAP for visualization
- **Clustering Algorithm**: K-means with optimal cluster determination
- **Similarity Metrics**: Cosine similarity for semantic matching

## Future Enhancements

- Integration with ticketing systems for real-time analysis
- Sentiment analysis for customer satisfaction tracking
- Automated escalation routing based on issue classification
- Multi-language support for global customer bases

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kaggle Customer Support Dataset](https://lnkd.in/dCgdzs6D) for providing the data
- Google Gemini API for enhanced natural language processing capabilities
- Streamlit for the interactive web application framework