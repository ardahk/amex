import streamlit as st
import pandas as pd
import os
import random
from PIL import Image

# Functions to get data from data frames
######################################
def get_all_orig_data():
    return pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/data/final_data.csv')

def get_products_by_id(user_id):
    all_data = get_all_orig_data()
    filter_data = all_data[all_data['user_id'] == user_id]
    final = filter_data[['name', 'num_of_item', 'category', 'department', 'cost']]
    return final

def get_random_user_id():
    df = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/final/users_final_data.csv')
    all_user_ids = df['user_id'].unique()
    random_id = random.choice(all_user_ids)
    
    # Check if the user ID exists in the data
    if df.loc[df['user_id'] == random_id].empty:
        return get_random_user_id()
    else:
        return random_id

def get_user_gender(user_id):
    df = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/final/users_final_data.csv')
    
    row = df.loc[df['user_id'] == user_id]
    if not row.empty:
        if row['gender_F'].iloc[0] == 1:
            return 'F'
        elif row['gender_M'].iloc[0] == 1:
            return 'M'
    return None 

def get_random_name(gender):
    df = pd.read_csv('data/user_imgs.csv')
    
    # Filter profiles by gender
    conditions = df['gender'] == gender
    filtered_profiles = df[conditions]
    
    # Check if there are any profiles that match the gender
    if filtered_profiles.empty:
        st.write(f"No profiles found for gender: {gender}")
        return None 
    
    return random.choice(filtered_profiles['name'].tolist())

def get_image_by_gender(gender):
    # Method only responsible for displaying the image
    user_profile = pd.read_csv('data/user_imgs.csv')
    filtered_profiles = user_profile[user_profile['gender'] == gender]
    
    random_profile = filtered_profiles.sample(n=1).iloc[0]
    
    # Load the image and link to user page
    image_path = f"user_imgs/{random_profile['img_url']}"
    
    return image_path