import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


property_df = pd.read_csv('property_df.csv')


st.title("Property Match Score Calculator")


st.subheader("Enter Your Preferences:")


budget = st.number_input("Your Budget", min_value=0, step=5000)
bedrooms = st.slider("Preferred Bedrooms", 1, 5, 3)
bathrooms = st.slider("Preferred Bathrooms", 1, 4, 2)

user_description = st.text_area("Enter Your Preferences Description")

st.subheader("Enter Property Characteristics:")

property_price = st.number_input("Property Price", min_value=0, step=1000)
property_bedrooms = st.slider("Property Bedrooms", 1, 5, 3)
property_bathrooms = st.slider("Property Bathrooms", 1, 4, 2)



property_description = st.text_area("Enter Property Description")


if st.button("Calculate Match Score"):
    
    if (budget == 0):
        st.error("Budget cannot be 0")
    if user_description == "" :
        st.error("Please enter your description of property!")
    if property_price == 0:
        st.error("Property Price cannot be 0")
    if property_description == "" : 
        st.error("Please enter description of property!")
    else :
        
        user_df = pd.DataFrame({
            'Budget': [budget],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'LivingArea': [0],
            'Price': [0]  # Placeholder for price, will not be used
        })

        user_scaled = user_df.copy()
        user_scaled[['Budget','Price','LivingArea', 'Bedrooms', 'Bathrooms' ]] = scaler.transform(user_df[['Budget','Price','LivingArea', 'Bedrooms', 'Bathrooms' ]])
        


        property_df_for_user = pd.DataFrame({
            'Budget': [0],  # Not needed for property, just placeholder
            'Price': [property_price],
            'Bedrooms': [property_bedrooms],
            'Bathrooms': [property_bathrooms],
            'LivingArea': [0]
        })
        # 'Budget', 'Price', 'LivingArea', 'Bedrooms', 'Bathrooms'
        property_scaled = property_df_for_user.copy()
        property_scaled[['Budget','Price','LivingArea', 'Bedrooms', 'Bathrooms']] = scaler.transform(property_df_for_user[['Budget','Price','LivingArea', 'Bedrooms', 'Bathrooms' ]])

        # Numerical similarity (only using the available numerical columns)
        numerical_similarity = cosine_similarity(user_scaled[[ 'Bedrooms', 'Bathrooms']],
                                                property_scaled[['Bedrooms', 'Bathrooms']])

        user_embedding = model.encode(user_description)
        property_embedding = model.encode(property_description)
        
        description_similarity = cosine_similarity([user_embedding], [property_embedding])
        
        
        budget_match = np.zeros((1, 1))
        user_budget_normalized = user_scaled['Budget'].values[0]
        property_price_normalized = property_scaled['Price'].values[0]
        
        if user_budget_normalized == 0:
            if property_price_normalized == 0:
                budget_match[0,0] = 1
            else :
                budget_match[0,0]= 0
        else:
            if property_price_normalized <= user_budget_normalized:
                budget_match[0, 0] = 1.0
            else:
                overBudget_ratio = property_price_normalized / user_budget_normalized
                budget_match[0, 0] = max(0, 1 - (overBudget_ratio - 1) / 2)
        
        
        match_weights = {'numerical': 0.4, 'description': 0.4, 'budget': 0.2}
        match_score = (
            match_weights['numerical'] * numerical_similarity +
            match_weights['description'] * description_similarity +
            match_weights['budget'] * budget_match
        )
        print(match_score)
        st.subheader(f"Match Score: {match_score[0][0]:.3f}")

        if match_score >= 0.5:
            st.write("This property is a **perfect Match** for you!")
        elif match_score > 0.4 and match_score < 0.5:
            st.write("This property is a **Good Match** for you! Although you might find some better ones!")   
        else:
            st.write("This property is **not a perfect match**, try adjusting your preferences!")

           

    