# importing the necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle


# reading cleaned data file
df = pd.read_csv("Cleaned_House_Rent_Train.csv")


# function to predict house rent price
def Price_prediction(type, latitude, longitude, property_size, total_floor, property_age, bathroom, lift, furnished):

    t = {"RK1": 1, "BHK1": 2, "BHK2": 3, "BHK3": 4, "BHK4": 5, "BHK4PLUS":6}
    type = t[type]

    l = {"Yes": 1, "No": 0}
    lift = l[lift]

    f = {"NOT_FURNISHED": 1, "SEMI_FURNISHED": 2, "FULLY_FURNISHED": 3}
    furnished = f[furnished]

    with open("house_rent_regression_model.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict([[type, latitude, longitude, property_size, total_floor, property_age, bathroom, lift, furnished]])

    return np.exp(prediction)



# streamlit setup
st.set_page_config("Smart Predicting Modelling for Rental Property Prices", layout = "wide")


selected = option_menu(None, 
                       options = ["Menu", "House Rent Prediction"],
                       icons = ["house"],
                       orientation = "horizontal",
                       styles = {"nav-link": {"font-size": "18px", "text-align": "center", "margin": "1px"},
                                 "icon": {"color": "yellow", "font-size": "20px"},
                                 "nav-link-selected": {"background-color": "#9457eb"}})


if selected == "Menu":
    
    st.title(":red[Smart Predictive Modeling for Rental Property Prices]")

    st.markdown("")

    st.markdown('''* In the real estate industry, determining the appropriate rental price for a property is crucial for
                   property owners, tenants, and property management companies. Accurate rent predictions can
                   help landlords set competitive prices, tenants make informed rental decisions, and property
                   management companies optimize their portfolio management.''')
    
    st.markdown('''* The goal of this project is to develop a data-driven model that predicts the rental price of
                   residential properties based on relevant features. By analyzing historical rental data and
                   property attributes, the model aims to provide accurate and reliable rent predictions.''')


if selected == "House Rent Prediction":

    with st.form("House Rent Prediction"):

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox(":blue[**Type**]", options = df["type"].unique(), key = "type")

            st.number_input(":blue[**Latitude**]", min_value = -90.0 ,max_value = 90.0, key = "lat")

            st.number_input(":blue[**Longitude**]", min_value = -180.0, max_value = 180.0, key = "lon")

            st.number_input(":blue[**Property Size**]", value = 1000, key = "ps")

            st.number_input(":blue[**Total Floor**]", value = 3, step = 1, key = "tf")

        with col2:
            st.number_input(":blue[**Property Age**]", value = 5, step = 1, key = "pa")

            st.number_input(":blue[**Bathroom**]", value = 2, key = "br")

            st.selectbox(":blue[**Furnishing**]", options = ["NOT_FURNISHED", "SEMI_FURNISHED", "FULLY_FURNISHED"], key = "fur")

            st.radio(":blue[**Lift**]", options = ["Yes", "No"], key = "lift" )

        
            if st.form_submit_button("**Predict**"):

                pred = Price_prediction(st.session_state["type"], st.session_state["lat"], st.session_state["lon"], 
                                        st.session_state["ps"], st.session_state["tf"], st.session_state["pa"], 
                                        st.session_state["br"], st.session_state["lift"], st.session_state["fur"])
                
                st.success(f"The Predicted House Rent Price is :green[₹ {pred[0]:,.0f}]")



# --------------------------x---------------------------------x-------------------------------------x------------------------------------x---------------------------------