import streamlit as st

def quiz():
    st.title("House Price Prediction Quiz")
    st.write("Test your knowledge with these quiz questions about house price prediction!")
    questions = [
        {
            "question": "What feature is most important for determining the price of a house?",
            "options": ["Roof Color", "Number of Bathrooms", "Location", "Number of Bedrooms"],
            "correct_answer": "Location"
        },
        {
            "question": "Which algorithm is commonly used for house price prediction?",
            "options": ["Naive Bayes", "K-means Clustering", "Decision Trees", "Linear Regression"],
            "correct_answer": "Linear Regression"
        },
        {
            "question": "What does 'BHK' stand for in real estate?",
            "options": ["Bedrooms, Hall, Kitchen", "Bathroom, Hall, Kitchen", "Big House, Kitchen", "Bedroom, Hall, Kid's Room"],
            "correct_answer": "Bedrooms, Hall, Kitchen"
        },
        {
            "question": "What is the typical unit of measurement for house area?",
            "options": ["Cubic Meters", "Square Feet", "Square Meters", "Hectares"],
            "correct_answer": "Square Feet"
        },
        {
            "question": "What is one common method used for handling missing data in house price prediction?",
            "options": ["Interpolation", "Extrapolation", "Imputation", "Deletion"],
            "correct_answer": "Imputation"
        },
        {
            "question": "What is the first step in building a house price prediction model?",
            "options": ["Model Training", "Feature Engineering", "Data Gathering", "Model Evaluation"],
            "correct_answer": "Data Gathering"
        },
        {
            "question": "What is the purpose of feature engineering in house price prediction?",
            "options": ["Reduce Overfitting", "Increase Data Size", "Improve Model Performance", "None of the above"],
            "correct_answer": "Improve Model Performance"
        },
        {
            "question": "Which of the following is not a factor that affects house prices?",
            "options": ["Size of the House", "Location", "Hair Color", "Economic Conditions"],
            "correct_answer": "Hair Color"
        },
        {
            "question": "What is the significance of regularization in linear regression for house price prediction?",
            "options": ["Reduce Bias", "Increase Model Complexity", "Prevent Overfitting", "None of the above"],
            "correct_answer": "Prevent Overfitting"
        },
        {
            "question": "What is the role of cross-validation in model evaluation for house price prediction?",
            "options": ["Select Features", "Assess Model Performance", "Improve Model Accuracy", "None of the above"],
            "correct_answer": "Assess Model Performance"
        }
    ]
    
    user_answers = {}
    for i, question_data in enumerate(questions):
        question = question_data["question"]
        options = question_data["options"]

        user_answer = st.radio(f"Question {i+1}: {question}", options, key=f"question_{i+1}")

        user_answers[f"user_answer_{i+1}"] = user_answer

    
    if st.button("Check Answers"):
        correct_answers = {f"user_answer_{i+1}": question_data["correct_answer"] for i, question_data in enumerate(questions)}
        for key, user_answer in user_answers.items():
            correct_answer = correct_answers[key]
            if user_answer.lower() == correct_answer.lower():
                st.success(f"Correct! 🎉")
            else:
                st.error(f"Incorrect! The correct answer is: {correct_answer}")
