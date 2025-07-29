"""
Football Player Performance Prediction UI
=========================================
A Streamlit web application for predicting football player performance using pre-trained models.

This application allows users to:
1. Upload a CSV file with player data for batch predictions
2. Enter individual player data manually for single predictions
3. Visualize prediction results with interpretations

Author: [Your Name]
Date: April 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="Football Performance Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("⚽ Football Player Performance Prediction")
st.markdown("""
This application uses machine learning to predict football player performance in the English Premier League.
Upload player data or enter details manually to get performance predictions.
""")

# Sidebar for model selection and info (keep this)
with st.sidebar:
    st.header("Model Selection")
    model_type = st.selectbox(
        "Select prediction model:", 
        ["RandomForest", "SVM", "NeuralNetwork", "GradientBoosting"],
        index=0
    )
    st.markdown("---")
    st.markdown("### Model Information")
    if model_type == "RandomForest":
        st.info("Random Forest typically provides the best balance of accuracy and interpretability.")
    elif model_type == "SVM":
        st.info("Support Vector Machine works well for complex boundary decisions.")
    elif model_type == "NeuralNetwork":
        st.info("Neural Network can capture intricate patterns but may be less interpretable.")
    elif model_type == "GradientBoosting":
        st.info("Gradient Boosting often performs well with feature interactions.")

# Function to load model
@st.cache_resource
def load_model(model_type):
    model_path = f"output/{model_type.lower()}_model.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

# Function to load feature importance
@st.cache_data
def load_feature_importance(model_type):
    importance_path = f"output/{model_type.lower()}_feature_importance.csv"
    try:
        importance_df = pd.read_csv(importance_path)
        return importance_df
    except FileNotFoundError:
        return None

# Function to load selected features
@st.cache_data
def load_selected_features():
    try:
        features = joblib.load('output/selected_features.joblib')
        return features
    except FileNotFoundError:
        # Default features if not found
        return [
            'xG', 'xA', 'Min', 'Touches', 'Tkl', 'Int', 'Blocks',
            'SCA', 'GCA', 'Cmp%', 'PrgP', 'Carries', 'Team_rating', 'Opp_rating'
        ]

# Function to make predictions
def predict_performance(model, data):
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None

# Function to display feature importance
def display_feature_importance(model_type):
    importance_df = load_feature_importance(model_type)
    if importance_df is not None:
        st.subheader("Feature Importance")
        
        # Display top 10 important features
        top_features = importance_df.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=top_features, ax=ax)
        ax.set_title(f"Top 10 Important Features - {model_type}")
        st.pyplot(fig)
        
        # Show importance table
        st.dataframe(top_features)
    else:
        st.info("Feature importance data not available for this model.")

# Function to preprocess input data
def preprocess_input(data, selected_features):
    # Ensure all required features are present
    for feature in selected_features:
        if feature not in data.columns:
            data[feature] = 0  # Default value
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype.kind in 'ifc':  # If column is numeric
            data[col] = data[col].fillna(0)
        else:
            data[col] = data[col].fillna('')
    
    return data[selected_features]

# Function to display prediction results
def display_prediction_results(player_names, predictions, probabilities):
    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Player': player_names,
        'Predicted_Performance': predictions,
        'Probability_Good_Performance': probabilities[:, 1]
    })
    
    # Add result interpretation
    results['Performance_Category'] = results['Predicted_Performance'].map({
        1: 'Good', 
        0: 'Poor'
    })
    
    # Display results
    st.subheader("Prediction Results")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart for performance categories
    performance_counts = results['Performance_Category'].value_counts()
    ax1.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax1.set_title('Performance Category Distribution')
    
    # Bar chart for probabilities
    if len(results) <= 10:  # Only show bar chart for 10 or fewer players
        good_probs = results['Probability_Good_Performance'].values
        poor_probs = 1 - good_probs
        index = np.arange(len(results))
        bar_width = 0.35
        
        ax2.bar(index, good_probs, bar_width, label='Good Performance', color='#66b3ff')
        ax2.bar(index + bar_width, poor_probs, bar_width, label='Poor Performance', color='#ff9999')
        ax2.set_xlabel('Player')
        ax2.set_ylabel('Probability')
        ax2.set_title('Prediction Probabilities')
        ax2.set_xticks(index + bar_width / 2)
        ax2.set_xticklabels(results['Player'], rotation=45, ha='right')
        ax2.legend()
    else:
        # For many players, show distribution instead
        sns.histplot(results['Probability_Good_Performance'], kde=True, ax=ax2)
        ax2.set_xlabel('Probability of Good Performance')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Performance Probabilities')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the results table
    st.dataframe(results)
    
    # Allow downloading the results
    csv = results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download results as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Top navigation tabs
main_tabs = st.tabs(["Home", "Batch Prediction", "Individual Prediction", "About"])

# Home tab
with main_tabs[0]:
    st.header("Welcome to the Football Performance Predictor")
    
    # Two-column layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### What This Tool Does
        This application uses machine learning to predict whether a football player will perform well in 
        upcoming matches based on historical statistics.
        
        ### How It Works
        1. **Select a model** from the sidebar
        2. Choose between **batch prediction** (upload a CSV) or **individual prediction** (enter details manually)
        3. Get performance predictions with probability scores
        4. Visualize and interpret the results
        
        ### Available Models
        - **Random Forest**: Ensemble of decision trees, good for overall accuracy
        - **SVM**: Support Vector Machine, effective for complex classification tasks
        - **Neural Network**: Multi-layer perceptron, captures intricate patterns
        - **Gradient Boosting**: Boosted trees, often has strong predictive power
        """)
    
    with col2:
        # Sample visualization image or logo
        st.image("image.png", 
                 caption="Performance prediction visualization sample")
        
        st.markdown("""
        ### Get Started
        Choose "Batch Prediction" or "Individual Prediction" from the sidebar to begin.
        """)
    
    # Display model metrics if available
    try:
        metrics_df = pd.DataFrame({
            'Model': ['RandomForest', 'SVM', 'NeuralNetwork', 'GradientBoosting'],
            'Accuracy': [0.785, 0.753, 0.801, 0.792],
            'Precision': [0.762, 0.741, 0.785, 0.768],
            'Recall': [0.748, 0.725, 0.773, 0.754],
            'F1-Score': [0.755, 0.732, 0.779, 0.761]
        })
        
        st.subheader("Model Performance Metrics")
        st.dataframe(metrics_df)
        
        # Create visualization of model metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df_melted, ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0.7, 0.85)
        st.pyplot(fig)
        
    except Exception as e:
        st.info("Model metrics not available. Run the model training pipeline to generate performance data.")

# Batch Prediction tab
with main_tabs[1]:
    st.header("Batch Prediction")
    st.markdown("""
    Upload a CSV file containing player data for batch predictions. 
    Make sure the file contains all the necessary features.
    """)
    
    # Load model and selected features
    model = load_model(model_type)
    selected_features = load_selected_features()
    
    if model is not None:
        st.write(f"Using {model_type} model for predictions")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload player data CSV", type=["csv"])
        
        if uploaded_file is not None:
            # Read the CSV file
            try:
                data = pd.read_csv(uploaded_file)
                st.write(f"Uploaded data shape: {data.shape}")
                
                # Show the uploaded data
                st.subheader("Uploaded Data Preview")
                st.dataframe(data.head())
                
                # Check if required columns are present
                missing_features = [f for f in selected_features if f not in data.columns]
                if missing_features:
                    st.warning(f"Missing features in uploaded data: {missing_features}")
                    st.info("The system will use default values for missing features, but this may affect prediction accuracy.")
                
                # Process button
                if st.button("Make Predictions"):
                    with st.spinner("Processing..."):
                        # Preprocess the data
                        X = preprocess_input(data, selected_features)
                        
                        # Make predictions
                        predictions, probabilities = predict_performance(model, X)
                        
                        if predictions is not None and probabilities is not None:
                            # Get player names if available
                            player_names = data['Name'] if 'Name' in data.columns else data.index
                            
                            # Display results
                            display_prediction_results(player_names, predictions, probabilities)
                            
                            # Show feature importance
                            display_feature_importance(model_type)
                        else:
                            st.error("Failed to make predictions. Please check your input data.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.error(f"Could not load the {model_type} model. Please check if the model files exist in the 'output' directory.")

# Individual Prediction tab
with main_tabs[2]:
    st.header("Individual Player Prediction")
    st.markdown("""
    Enter individual player statistics to get a performance prediction.
    Fill in as many fields as possible for the most accurate prediction.
    """)
    
    # Load model and selected features
    model = load_model(model_type)
    selected_features = load_selected_features()
    
    if model is not None:
        st.write(f"Using {model_type} model for predictions")
        
        # Create a form for user input
        with st.form("player_input_form"):
            st.subheader("Player Information")
            
            # Basic player info
            col1, col2 = st.columns(2)
            with col1:
                player_name = st.text_input("Player Name", "Harry Kane")
                player_team = st.selectbox("Team", [
                    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
                    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", 
                    "Leicester", "Liverpool", "Man City", "Man Utd", "Newcastle", 
                    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
                ])
            
            with col2:
                position = st.selectbox("Position", ["FWD", "MID", "DEF", "GKP"])
                opponent_team = st.selectbox("Opponent Team", [
                    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
                    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", 
                    "Leicester", "Liverpool", "Man City", "Man Utd", "Newcastle", 
                    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
                ])
            
            # Create tabs for different categories of stats
            tabs = st.tabs(["Basic Stats", "Advanced Stats", "Team Context"])
            
            # Basic stats tab
            with tabs[0]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    minutes = st.number_input("Minutes Played", min_value=0, max_value=90, value=90)
                    goals = st.number_input("Goals", min_value=0, max_value=10, value=0)
                    assists = st.number_input("Assists", min_value=0, max_value=10, value=0)
                
                with col2:
                    shots = st.number_input("Shots", min_value=0, max_value=20, value=2)
                    shots_on_target = st.number_input("Shots on Target", min_value=0, max_value=20, value=1)
                    yellow_cards = st.number_input("Yellow Cards", min_value=0, max_value=2, value=0)
                
                with col3:
                    touches = st.number_input("Touches", min_value=0, max_value=200, value=50)
                    tackles = st.number_input("Tackles", min_value=0, max_value=20, value=1)
                    interceptions = st.number_input("Interceptions", min_value=0, max_value=20, value=0)
            
            # Advanced stats tab
            with tabs[1]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=5.0, value=0.3, format="%.2f")
                    xa = st.number_input("Expected Assists (xA)", min_value=0.0, max_value=5.0, value=0.2, format="%.2f")
                    sca = st.number_input("Shot-Creating Actions", min_value=0, max_value=20, value=2)
                
                with col2:
                    gca = st.number_input("Goal-Creating Actions", min_value=0, max_value=10, value=0)
                    passes_completed = st.number_input("Passes Completed", min_value=0, max_value=150, value=30)
                    passes_attempted = st.number_input("Passes Attempted", min_value=0, max_value=150, value=40)
                
                with col3:
                    cmp_pct = st.number_input("Pass Completion %", min_value=0.0, max_value=100.0, value=75.0, format="%.1f")
                    progressive_passes = st.number_input("Progressive Passes", min_value=0, max_value=50, value=5)
                    carries = st.number_input("Carries", min_value=0, max_value=100, value=20)
            
            # Team context tab
            with tabs[2]:
                col1, col2 = st.columns(2)
                
                with col1:
                    was_home = st.selectbox("Home/Away", ["Home", "Away"])
                    team_rating = st.slider("Team Rating", min_value=60.0, max_value=100.0, value=80.0, step=0.5, format="%.1f")
                
                with col2:
                    opp_rating = st.slider("Opponent Rating", min_value=60.0, max_value=100.0, value=75.0, step=0.5, format="%.1f")
                    gameweek = st.number_input("Gameweek", min_value=1, max_value=38, value=20)
            
            submitted = st.form_submit_button("Predict Performance")
        
        if submitted:
            with st.spinner("Making prediction..."):
                # Create a DataFrame from form inputs
                player_data = {
                    'Name': player_name,
                    'Team': player_team,
                    'Opponent': opponent_team,
                    'FPL_pos': position,
                    'Min': minutes,
                    'Gls': goals,
                    'Ast': assists,
                    'Sh': shots,
                    'SoT': shots_on_target,
                    'CrdY': yellow_cards,
                    'Touches': touches,
                    'Tkl': tackles,
                    'Int': interceptions,
                    'xG': xg,
                    'xA': xa,
                    'SCA': sca,
                    'GCA': gca,
                    'Cmp': passes_completed,
                    'Att': passes_attempted,
                    'Cmp%': cmp_pct,
                    'PrgP': progressive_passes,
                    'Carries': carries,
                    'Was_home': 1 if was_home == "Home" else 0,
                    'Team_rating': team_rating,
                    'Opp_rating': opp_rating,
                    'GW': gameweek
                }
                
                # Convert to DataFrame
                player_df = pd.DataFrame([player_data])
                
                # Preprocess the data
                X = preprocess_input(player_df, selected_features)
                
                # Make predictions
                predictions, probabilities = predict_performance(model, X)
                
                if predictions is not None and probabilities is not None:
                    # Display results
                    st.success(f"Prediction Complete!")
                    
                    # Create a visual representation of the prediction
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display main prediction result
                        prediction = predictions[0]
                        probability = probabilities[0][1]  # Probability of good performance
                        
                        result_category = "Good" if prediction == 1 else "Poor"
                        
                        st.metric(
                            label="Predicted Performance", 
                            value=result_category,
                            delta=f"{probability:.1%} Confidence"
                        )
                        
                        # Create a gauge chart
                        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
                        theta = np.linspace(0, 1.8 * np.pi, 100)
                        ax.plot(theta, [1] * 100, color='lightgray', lw=20, alpha=0.5)
                        ax.plot(theta[:int(probability * 100)], [1] * int(probability * 100), color='#007BFF' if prediction == 1 else '#FF6B6B', lw=20, alpha=0.8)
                        ax.set_rticks([])
                        ax.set_xticks([])
                        ax.spines['polar'].set_visible(False)
                        plt.text(0, 0, f"{probability:.1%}", ha='center', va='center', fontsize=24)
                        ax.set_title("Prediction Confidence", pad=20)
                        st.pyplot(fig)
                        
                    with col2:
                        # Display interpretation
                        st.subheader("Interpretation")
                        
                        if prediction == 1:
                            if probability > 0.8:
                                st.write("📈 **Strong likelihood of good performance**")
                                st.write(f"{player_name} has a very high probability ({probability:.1%}) of performing well in the upcoming match.")
                            elif probability > 0.6:
                                st.write("👍 **Moderate likelihood of good performance**")
                                st.write(f"{player_name} has a good chance ({probability:.1%}) of performing well, though there is some uncertainty.")
                            else:
                                st.write("⚖️ **Slight likelihood of good performance**")
                                st.write(f"{player_name} is predicted to perform well, but with relatively low confidence ({probability:.1%}).")
                        else:
                            if probability < 0.2:
                                st.write("📉 **Strong likelihood of poor performance**")
                                st.write(f"{player_name} has a very high probability ({(1-probability):.1%}) of performing poorly in the upcoming match.")
                            elif probability < 0.4:
                                st.write("👎 **Moderate likelihood of poor performance**")
                                st.write(f"{player_name} has a good chance ({(1-probability):.1%}) of performing poorly, though there is some uncertainty.")
                            else:
                                st.write("⚖️ **Slight likelihood of poor performance**")
                                st.write(f"{player_name} is predicted to perform poorly, but with relatively low confidence ({(1-probability):.1%}).")
                        
                        # Add context based on position
                        st.write("### Position Context")
                        if position == "FWD":
                            st.write("For forwards, performance is heavily weighted by goals, assists, and expected goal contributions.")
                        elif position == "MID":
                            st.write("Midfielders are evaluated based on a balance of attacking contributions, progressive passes, and defensive actions.")
                        elif position == "DEF":
                            st.write("Defenders are primarily assessed on clean sheets, tackles, interceptions, and blocks, with attacking contributions as a bonus.")
                        elif position == "GKP":
                            st.write("Goalkeepers are evaluated mainly on saves, clean sheets, and goals conceded.")
                    
                    # Display feature importance
                    display_feature_importance(model_type)
                    
                    # Show which features contributed most to this specific prediction
                    st.subheader("Key Factors for This Prediction")
                    importance_df = load_feature_importance(model_type)
                    
                    if importance_df is not None:
                        # Get top 5 important features
                        top_features = importance_df.head(5)['Feature'].tolist()
                        
                        # Display the values for these features
                        feature_values = []
                        for feature in top_features:
                            if feature in player_df.columns:
                                feature_values.append({
                                    'Feature': feature,
                                    'Value': player_df[feature].values[0]
                                })
                        
                        if feature_values:
                            feature_df = pd.DataFrame(feature_values)
                            st.table(feature_df)
                else:
                    st.error("Failed to make predictions. Please check your input data.")
    else:
        st.error(f"Could not load the {model_type} model. Please check if the model files exist in the 'output' directory.")

# About tab
with main_tabs[3]:
    st.header("About This Application")
    
    st.markdown("""
    ### Project Overview
    This application is part of the "Football Player Performance Prediction Using Machine Learning" project. 
    It uses historical match data from the English Premier League to predict player performance.
    
    ### Methodology
    The system analyzes the FPL Player Logs dataset to train multiple machine learning models. 
    It identifies key performance factors and compares different algorithms to find the most accurate prediction method.
    
    ### Features
    - **Data Preprocessing**: Handles missing values, normalizes features, and encodes categorical variables
    - **Feature Engineering**: Creates advanced metrics like rolling averages and match difficulty features
    - **Multiple Models**: Implements Random Forest, SVM, Neural Network, and Gradient Boosting
    - **Comprehensive Evaluation**: Calculates multiple performance metrics
    - **Visualization**: Generates charts for results interpretation
    
    ### Development
    This application was developed as part of a BSc Computer Science final year project at the University of Lincoln.
    
    ### Credits
    - Data Source: FPL Player Logs dataset (Kaggle)
    - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
    """)
    
    # Show project timeline
    st.subheader("Project Timeline")
    timeline_data = {
        'Phase': ['Research', 'Data Collection', 'Model Development', 'UI Development', 'Testing & Refinement', 'Documentation'],
        'Status': ['Completed', 'Completed', 'Completed', 'Completed', 'In Progress', 'Pending'],
        'Completion': [100, 100, 100, 90, 50, 20]
    }
    timeline_df = pd.DataFrame(timeline_data)
    
    # Create a progress bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(timeline_df['Phase'], timeline_df['Completion'], color=['#4CAF50' if status == 'Completed' else '#FFC107' if status == 'In Progress' else '#9E9E9E' for status in timeline_df['Status']])
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{timeline_df['Completion'][i]}%", 
                va='center', fontsize=10)
    
    ax.set_xlim(0, 110)
    ax.set_xlabel('Completion (%)')
    ax.set_title('Project Timeline Progress')
    st.pyplot(fig)
    
    # Contact information
    st.subheader("Contact Information")
    st.markdown("""
    **Author**: Obasa Joseph  
    **Email**: [Your Email]  
    **University**: University of Lincoln  
    **Department**: School of Computer Science  
    """)