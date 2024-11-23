import streamlit as st
import pandas as pd
import os
import random
from PIL import Image

# Functions to get Data
######################################
def get_all_orig_data():
    return pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/data/final_data.csv')

def get_products_by_user_id(user_id):
    df = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/data/final_data.csv')
    filtered_products = df.loc[df['user_id'] == user_id, 'name']
    return filtered_products

def get_random_user_id():
    df = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/data/final_data.csv')
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
