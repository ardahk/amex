import streamlit as st
from PIL import Image
from load_data import *

# Display User Profiles and Products
######################################
def display_user_profile(gender):
    user_profile = pd.read_csv('data/user_imgs.csv')
    filtered_profiles = user_profile[user_profile['gender'] == gender]
    
    if not user_profile.empty:
        random_profile = filtered_profiles.sample(n=1).iloc[0]
        # Get name
        st.header(random_profile['name'])
        
        # Attempt to load the image and handle potential errors
        image_path = f"user_imgs/{random_profile['img_url']}"
        image = Image.open(image_path)
        st.image(image)
    else:
        st.write("No profiles found for this gender.")

def display_user_products(user_id):
    # Fetch all original data
    user_data = get_all_orig_data()
    
    if not user_data.empty:
        st.subheader("User Products:")
        # Filter products by user ID
        user_products = user_data[user_data['user_id'] == user_id]
        display_columns = ['name','num_of_item','category', 'department', 'cost']
        products = user_products[display_columns]
        st.markdown(products.to_html(index=False), unsafe_allow_html=True)
    else:
        st.write("No data available.")



