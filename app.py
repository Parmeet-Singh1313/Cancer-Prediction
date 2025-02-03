# cancer_prediction.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    data = pd.read_csv('cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    return data

@st.cache_resource
def train_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred, y_proba

# Streamlit app
def main():
    st.title("🩺 Advanced Cancer Prediction System")

    data = load_data()

    st.sidebar.header("Patient Features")
    input_features = []
    

    for col in data.columns[1:]:  # Use all features except diagnosis
        min_val = float(data[col].min())
        max_val = float(data[col].max())
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
    
    if st.sidebar.button("Predict"):
        input_array = np.array(input_features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        
        st.subheader("Prediction Result")
        result = "Malignant (Cancer)" if prediction[0] == 1 else "Benign (No Cancer)"
        
        # Change color based on prediction
        if prediction[0] == 1:
            st.error(f"**Diagnosis:** {result}")
        else:
            st.success(f"**Diagnosis:** {result}")
            
        st.info(f"Model Accuracy: {accuracy:.2%}")
    

    st.subheader("Data Analysis")
    
 
    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
   
    with st.expander("Feature Importance"):
        feature_importance = pd.DataFrame({
            'Feature': data.columns[1:],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis', ax=ax2)
        st.pyplot(fig2)
  
    with st.expander("Feature Distributions"):
        selected_feature = st.selectbox("Select feature to view distribution:", data.columns[1:])
        fig3, ax3 = plt.subplots()
        sns.histplot(data=data, x=selected_feature, hue='diagnosis', kde=True, element='step', palette='viridis')
        st.pyplot(fig3)
    

    st.subheader("Model Evaluation")
    
 
    with st.expander("Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig4, ax4 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_xticklabels(['Benign', 'Malignant'])
        ax4.set_yticklabels(['Benign', 'Malignant'])
        st.pyplot(fig4)
 
    with st.expander("ROC Curve"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig5, ax5 = plt.subplots()
        ax5.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax5.plot([0, 1], [0, 1], 'k--')
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curve')
        ax5.legend()
        st.pyplot(fig5)
    
    with st.expander("Decision Tree Structure"):
        fig6, ax6 = plt.subplots(figsize=(20, 10))
        plot_tree(model, 
                 feature_names=data.columns[1:], 
                 class_names=['Benign', 'Malignant'], 
                 filled=True, 
                 ax=ax6, 
                 max_depth=3,
                 proportion=True,
                 rounded=True)
        st.pyplot(fig6)

if __name__ == '__main__':
    main()