import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px 

house_price_data = pd.read_csv("pune.csv")


data = pd.DataFrame({
    'Area': ['Aundh', 'Baner', 'Kothrud', 'Kalyani Nagar', 'Wakad'],
    'Latitude': [18.5544, 18.5482, 18.5096, 18.5485, 18.5919],
    'Longitude': [73.8127, 73.7732, 73.8074, 73.9023, 73.7627],
    'House_Price (INR)': [7500000, 8500000, 6800000, 9500000, 7200000]
})


def visualize(data):
    fig = px.scatter_mapbox(data, lat="Latitude", lon="Longitude", hover_name="Area",
                            hover_data=["House_Price (INR)"],
                            color="House_Price (INR)",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            zoom=10)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)


def visualize_temporal_trends(data):
    st.header("Temporal analysis of House prices in Pune")
    data['Synthetic_Time'] = range(len(data))

    average_prices = data.groupby('Synthetic_Time')['price'].mean().reset_index()

    fig = px.line(average_prices, x='Synthetic_Time', y='price', title='Average House Prices Over Time')
    fig.update_xaxes(title='Synthetic Time')
    fig.update_yaxes(title='Average House Price')
    st.plotly_chart(fig)
