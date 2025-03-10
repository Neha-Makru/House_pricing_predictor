import streamlit as st 

def faq_page():
    st.title("FAQ Page")
    st.write("*Welcome to the FAQ Page!*")
    st.write("Here, you can find answers to frequently asked questions about house price prediction models.")
    st.write("*Q: What is house price prediction?*")
    st.write("A: House price prediction is the task of using machine learning algorithms to estimate the value of residential properties based on various features.")
    st.write("*Q: Which factors influence house prices?*")
    st.write("A: Factors include location, size, number of bedrooms/bathrooms, amenities, neighborhood characteristics, and economic trends.")
    st.write("*Q: How accurate are house price prediction models?*")
    st.write("A: Accuracy varies based on data quality, feature selection, model complexity, and the predictive power of the chosen algorithm.")
    st.write("*Q: How do feature importance and model evaluation help in house price prediction?*")
    st.write("A: Feature importance analysis identifies which features have the greatest impact on predictions. Model evaluation metrics assess the performance and reliability of the model.")
    st.write("*Q: What are some limitations of house price prediction models?*")
    st.write("A: Limitations include data availability, model assumptions, sensitivity to outliers, and the dynamic nature of housing markets.")
