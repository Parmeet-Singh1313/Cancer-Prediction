import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Enhanced Caching and Error Handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('cancer.csv')
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

@st.cache_resource
def train_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred, y_proba

def main():
    st.set_page_config(page_title="Cancer Prediction System", page_icon="🩺", layout="wide")
    
    st.title("🩺 Advanced Cancer Prediction System")
    st.markdown("### AI-Powered Diagnostic Support Tool")

    data = load_data()
    if data is None:
        st.stop()

    # AI Risk Insights Section
    st.sidebar.header("🔬 Patient Risk Assessment")
    input_features = []

    # Dynamic Feature Input with Risk Indicators
    risk_guidance = {
        'low_risk': '✅ Low Risk',
        'moderate_risk': '⚠️ Moderate Risk',
        'high_risk': '🚨 High Risk'
    }

    for col in data.columns[1:]:
        min_val, max_val = float(data[col].min()), float(data[col].max())
        default_val = float(data[col].median())
        input_val = st.sidebar.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val)/100
        )
        input_features.append(input_val)

    model, scaler, accuracy, X_test, y_test, y_pred, y_proba = train_model(data)
    
    # Prediction and Risk Assessment
    if st.sidebar.button("Predict Risk"):
        input_array = np.array(input_features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        risk_prob = model.predict_proba(scaled_input)[0][1]

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", "Malignant" if prediction[0] == 1 else "Benign")
        
        with col2:
            st.metric("Risk Probability", f"{risk_prob*100:.2f}%")
        
        with col3:
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Risk Level Color Coding
        risk_color = {
            0: "green", 
            1: "red"
        }
        
        st.markdown(f"### Risk Assessment: {'🚨 High Risk' if prediction[0] == 1 else '✅ Low Risk'}")
        st.progress(risk_prob)

    # Advanced Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Feature Analysis", "📊 Model Performance", "🌳 Decision Tree", "📈 Comprehensive Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            fig1 = px.imshow(data.corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': data.columns[1:],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig2 = px.bar(
                feature_importance.head(10), 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Top 10 Most Important Features"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig3 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
            fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
            
            fig4.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title='Receiver Operating Characteristic (ROC) Curve'
            )
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Decision Tree Visualization")
        fig5, ax5 = plt.subplots(figsize=(20, 10))
        plot_tree(model, 
                  feature_names=data.columns[1:].tolist(), 
                  class_names=['Benign', 'Malignant'],
                  filled=True, 
                  ax=ax5, 
                  max_depth=3,
                  proportion=True,
                  rounded=True)
        st.pyplot(fig5)

    with tab4:
        st.header("📊 Feature Distribution Analysis")
    
        # Feature Selection Dropdown
        selected_feature = st.selectbox(
            "Select Feature to Visualize", 
            data.columns[1:],
            key="feature_distribution_main"
        )
        
        # Create two columns for benign and malignant distributions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Benign Tumor Distribution")
            benign_dist = px.histogram(
                data[data['diagnosis'] == 0], 
                x=selected_feature,
                title=f"Benign {selected_feature} Distribution",
                color_discrete_sequence=['blue']
            )
            st.plotly_chart(benign_dist, use_container_width=True)
        
        with col2:
            st.subheader("Malignant Tumor Distribution")
            malignant_dist = px.histogram(
                data[data['diagnosis'] == 1], 
                x=selected_feature,
                title=f"Malignant {selected_feature} Distribution",
                color_discrete_sequence=['red']
            )
            st.plotly_chart(malignant_dist, use_container_width=True)
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Benign Mean", 
                    f"{data[data['diagnosis']==0][selected_feature].mean():.2f}")
        with col2:
            st.metric("Malignant Mean", 
                    f"{data[data['diagnosis']==1][selected_feature].mean():.2f}")
        st.subheader("Feature Relationship Visualization")
        
        # Pairwise Feature Selection
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox(
                "X-axis Feature", 
                data.columns[1:],
                key="pairwise_x_selector"
            )
        
        with col2:
            feature_y = st.selectbox(
                "Y-axis Feature", 
                data.columns[1:],
                key="pairwise_y_selector"
            )
        
        # Scatter Plot with Diagnosis Coloring
        fig_scatter = px.scatter(
            data, 
            x=feature_x, 
            y=feature_y, 
            color='diagnosis',
            color_discrete_map={0: 'blue', 1: 'red'},
            title=f"{feature_x} vs {feature_y} by Diagnosis",
            labels={'diagnosis': 'Tumor Type'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == '__main__':
    main()
