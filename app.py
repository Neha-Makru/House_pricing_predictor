import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from Estimate import prediction, locations
from Visualization import visualize, visualize_temporal_trends, data
from FAQ import faq_page
from Quiz import quiz
from Educational import edu, data_gathering, data_preprocessing, feature_engineering, model_building, model_evaluation

house_price_data = pd.read_csv("pune.csv")

def predict():
    location = st.selectbox("Select the Location please", locations)
    bhk = st.text_input("Enter BHK in numbers", "")
    sqft = st.text_input("Enter total house area in sqft", "")
    bath = st.text_input("Enter number of bathroom(s) in numbers", "")
    balcony = st.text_input("Enter number of balcony(ies) in numbers", "")
    area_type = st.selectbox('Area Type', ['Built-up  Area', 'Super built-up  Area'])
    availability = st.selectbox('Availability', ['Ready To Move', 'Not Ready'])

    if st.button('Predict Price'):
        result = prediction(location, bhk, bath, balcony, sqft, area_type, availability)
        st.success(f'The estimated price for the property is Rs. {result:.2f} Lakhs.')
    else:
        st.warning("Please fill in all the input fields.")
    
def main():
    
    page = st.sidebar.radio("Navigation, Go to :compass:", ["Predict my price", "Educational", 
                                           "Visualization","Quiz Mania","FAQ"], 
                             format_func=lambda x: f"{x} {':house_with_garden:' if x == 'Predict my price' else ''}"
                                                  f"{':books:' if x == 'Educational' else ''}"
                                                  f"{':chart_with_upwards_trend:' if x == 'Visualization' else ''}"
                                                  f"{':question:' if x == 'FAQ' else ''}"
                                                  f"{':trophy:' if x == 'Quiz Mania' else ''}")
    st.title('ML Property Valuer')
    st.write('''
            ### This app predicts the values of the different **house prices** in Pune. 
            ''')
    st.markdown("---")
    
    if page=="Educational":
        edu()
        data_gathering()
        data_preprocessing()
        feature_engineering()
        model_building()
        model_evaluation()
    elif page=="Predict my price":
        predict()
    elif page == "FAQ":
        faq_page()
    elif page == "Quiz Mania":
        quiz()
    elif page == "Visualization":
        visualize(data)
        visualize_temporal_trends(house_price_data)
      
if __name__ == "__main__":
    main()
