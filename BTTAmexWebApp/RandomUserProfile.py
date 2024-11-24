import streamlit as st
import pandas as pd
from load_data import *
from recommendation_model import *
import json

class RandomUserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.gender = get_user_gender(self.user_id)
        self.name = get_random_name(self.gender)
        self.image = get_image_by_gender(self.gender)  
        self.products = get_products_by_id(self.user_id)
    
    def to_dict(self):
        return {
            "user_id": int(self.user_id),  # Ensure user_id is an integer
            "gender": self.gender,
            "name": self.name,
            "image": self.image,  # Store the image URL
            "products": self.products.to_dict() if isinstance(self.products, pd.DataFrame) else self.products
        }
        
    @classmethod
    def from_dict(cls, data):
        obj = cls(data['user_id'])
        obj.gender = data['gender']
        obj.name = data['name']
        obj.image = data['image']  # Set the image URL
        obj.products = data['products']
        return obj
    
    def get_user_id(self):
        return self.user_id
    
    def get_name(self):
        return self.name
    
    def get_gender(self):
        return self.gender
    
    def get_user_products(self):
        return self.products
    
    def get_user_image(self):
        return self.image
        
    def display_user_products(self):
        # Fetch all original data
        user_data = get_all_orig_data()
        
        if not user_data.empty:
            st.subheader("User Products:")
            # Filter products by user ID
            user_products = user_data[user_data['user_id'] == self.user_id]
            display_columns = ['name', 'num_of_item', 'category', 'department', 'cost']
            products = user_products[display_columns]
            st.markdown(products.to_html(index=False), unsafe_allow_html=True)
        else:
            st.write("No data available.")
        
    def display_user_recommendations(self, top_n):
        model = build_model()
        generate_recommendations_for_user(self.user_id, model, top_n)

    

