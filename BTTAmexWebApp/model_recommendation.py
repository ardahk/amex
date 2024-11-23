import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dot, BatchNormalization
from tensorflow.keras.models import Model

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

def generate_recommendations_for_user(user_id, model, top_n=1):
    user_row = users_final[users_final['user_id'] == user_id]
    if user_row.empty:
        return []

    # prepare user data
    user_data = user_row.drop(columns=['product_id', 'user_id']).values
    user_data_repeated = np.repeat(user_data, len(products_final), axis=0)

    # prepare product data
    product_data = products_final.drop(columns=['product_id', 'flattened_name_embedding', 'flattened_brand_embedding']).values

    # predict probabilities
    predicted_probabilities = model.predict([user_data_repeated, product_data]).flatten()

    # sort products by descending probabilities
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    top_recommendations = products_final.iloc[sorted_indices[:top_n]]['product_id'].tolist()

    # map product IDs to names
    product_names = original_products[original_products['id'].isin(top_recommendations)]['name'].tolist()
    return product_names

