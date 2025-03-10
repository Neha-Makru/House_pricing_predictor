import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def edu():
    st.write("*Here, we are going to look at a brief summary regarding what happens in the house price prediction Machine learning model*")
    st.image("house.png",caption="House price prediction", use_column_width=True)
    st.markdown("---")
    st.write("*So here are the basic steps involved in the model building for the house price estimations.*")
    
def data_gathering():
    with st.expander("Step 1: Data Gathering and Data Collection"):
        st.image("Step_1.png", caption="Steps involved in the data collection process", use_column_width=True)
        st.write("In this step, relevant data about houses is collected. "
                 "This data can include information about the area, number of bedrooms, "
                 "bathrooms, square footage, location, etc.")
        st.subheader("Data Distribution")
        data = [10, 15, 20, 25, 30, 35, 40]
        fig, ax = plt.subplots()
        ax.hist(data, bins=5)
        st.pyplot(fig)

def data_preprocessing():
    with st.expander("Step 2: Data Preprocessing"):
        st.write("Data preprocessing involves cleaning and preparing the collected data "
                 "for analysis. This includes handling missing values, removing duplicates, "
                 "and transforming data into a suitable format.")
        st.subheader("Handling Missing Values")
        data = {'A': [1, 2, None, 4, 5], 'B': [None, 2, 3, 4, 5]}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

def feature_engineering():
    with st.expander("Step 3: Feature Engineering"):
        st.write("Feature engineering involves creating new features or transforming existing ones "
                 "to improve the performance of the machine learning model. This can include feature scaling, "
                 "encoding categorical variables, or creating interaction terms.")
        st.subheader("Feature Importance")
        features = ['Area', 'Bedrooms', 'Bathrooms', 'Location']
        importance = [0.5, 0.3, 0.2, 0.1]
        fig, ax = plt.subplots()
        ax.bar(features, importance)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)


def model_building():
    with st.expander("Step 4: Model Building"):
        st.write("In this step, a machine learning model is trained on the preprocessed data. "
                 "Common algorithms used for house price prediction include linear regression, "
                 "decision trees, random forests, and gradient boosting.")
        
        st.subheader("Model Performance")
        models = ['Linear Regression', 'Decision Trees', 'Random Forests', 'Gradient Boosting']
        accuracy = [0.75, 0.80, 0.85, 0.90]
        fig, ax = plt.subplots()
        ax.bar(models, accuracy, color='green')
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance')
        st.pyplot(fig)

def model_evaluation():
    with st.expander("Step 5: Model Evaluation"):
        st.write("Once the model is trained, it is evaluated using metrics such as mean squared error (MSE), "
                 "R-squared score, or mean absolute error (MAE). This step helps assess the performance "
                 "of the model and identify areas for improvement.")
    
        st.subheader("Model Evaluation")
        metrics = ['MSE', 'R-squared', 'MAE']
        values = [10, 0.85, 5]
        fig, ax = plt.subplots()
        ax.bar(metrics, values, color='orange')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Model Evaluation')
        st.pyplot(fig)
        
        st.write("""*In conclusion, the processes involved in the process of house price 
                 prediction are illustrated in the diagram given below*""")
