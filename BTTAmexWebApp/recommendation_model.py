import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dot, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler


users_final = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/final/users_final_data.csv')
products_final = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/final/products_final_data.csv')
original_products = pd.read_csv('https://raw.githubusercontent.com/ardahk/amex/refs/heads/main/data/products.csv')

# Asmi's Code with slight modifications
def build_model():
    user_input = Input(shape=(16,), name='user_input')
    item_input = Input(shape=(30,), name='item_input')

    user_tower = Dense(128, activation='relu')(user_input)
    user_tower = BatchNormalization()(user_tower)

    item_tower = Dense(128, activation='relu')(item_input)
    item_tower = BatchNormalization()(item_tower)

    dot_product = Dot(axes=1)([user_tower, item_tower])
    model = Model(inputs=[user_input, item_input], outputs=dot_product)
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_recommendations_for_user(user_id, model, top_n=5):
    # Get the user data for the given user_id
    user_row = users_final[users_final['user_id'] == user_id]
    if user_row.empty:
        st.warning(f"User ID {user_id} not found.")
        return []

    # Normalize the user and product data
    user_scaler = StandardScaler()
    product_scaler = StandardScaler()

    # Normalize user data (drop product_id and user_id first)
    user_data_normalized = user_scaler.fit_transform(user_row.drop(columns=['product_id', 'user_id']).values)

    # Normalize product data (drop irrelevant columns)
    product_data_normalized = product_scaler.fit_transform(products_final.drop(columns=['product_id', 'flattened_name_embedding', 'flattened_brand_embedding']).values)

    # Prepare normalized user data and repeat it for each product
    user_data_repeated = np.repeat(user_data_normalized, len(products_final), axis=0)

    # Repeat the product data to match the number of repeated user data rows
    product_data_repeated = np.tile(product_data_normalized, (len(user_data_normalized), 1))

    # Predict probabilities using the model
    predicted_probabilities = model.predict([user_data_repeated, product_data_repeated]).flatten()

    # Sort products by descending probabilities
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    
    # Ensure that indices are within bounds of the products_final DataFrame
    valid_indices = sorted_indices[:top_n]
    valid_indices = valid_indices[valid_indices < len(products_final)]  # Check bounds
    
    # If there are not enough valid recommendations, reduce top_n to valid length
    top_n = len(valid_indices)
    
    # Get the top product IDs from the sorted indices
    top_recommendations = products_final.iloc[valid_indices]['product_id'].tolist()

    # Map product IDs to product names
    product_names = original_products[original_products['id'].isin(top_recommendations)]['name'].tolist()

    return product_names





