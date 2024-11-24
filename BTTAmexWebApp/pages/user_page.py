import streamlit as st
import json
import pandas as pd
from recommendation_model import *

st.set_page_config(
    page_title="User Page", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# Link to another Streamlit page in the same tab
st.markdown(
    '<a href="./" target="_self" style="text-decoration:none;"><h5>⬅️ Go Back to Home Page</h5></a>',
    unsafe_allow_html=True,
)

# Get query parameters from the URL
query_params = st.query_params

# Retrieve the 'user_id' from the query parameters
user_id = int(query_params.get("user_id", [None]))

json_file_path = "./user_profiles.json"

# Load the JSON file
try:
    with open(json_file_path, "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File not found: {json_file_path}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")

user = None

# Find the user with the given user_id
for i in range(len(data)):
    if data[i]['user_id'] == int(user_id):
        user = data[i]
        break
                
# Display user profile
if user is not None:
    products = pd.DataFrame(user['products'])
    st.title(user['name'])
    st.subheader("User_id: " + str(user['user_id']))
    st.image(user['image'])
    st.table(products)

# Test the model's ability to generate recommendations
model = build_model()

# Simulate a simple recommendation for debugging
recommendations = generate_recommendations_for_user(user_id, model)

st.header("Recommendations")
df_recommendations = pd.DataFrame(recommendations, columns=["Product Name"])

# Display the DataFrame without the index using st.table()
st.markdown(df_recommendations.to_html(index=False), unsafe_allow_html=True)
