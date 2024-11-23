from display_functions import *
import streamlit as st
from load_data import *
from model_recommendation import *

class Home:
    # Load Home Page
    @staticmethod
    def load_home_page():
        st.subheader("BTT AI Studio and Amex Team #1 Presents:")
        st.header("User Product Recommendation System")
        st.write("This Application Demonstrates our Two-Tower Product Recommendation Model.")
        st.write("---")

        # Load model
        model = build_model()

        # Display profiles and recommendations for 6 random users
        for _ in range(6):
            df = get_all_orig_data()

            # Get a random user ID and their gender
            user_id = get_random_user_id()
            if user_id is None:
                st.error("Could not find a valid user ID.")
                continue

            user_gender = get_user_gender(user_id)
            if user_gender is None:
                st.error(f"Could not retrieve gender for user ID: {user_id}")
                continue

            # Display user profile and products
            display_user_profile(user_gender)
            display_user_products(user_id)

            # Generate and display recommendations
            recommendations = generate_recommendations_for_user(user_id, model)
            if recommendations:
                st.subheader("Recommendations:")
                for product in recommendations:
                    st.write(f"- {product}")
            else:
                st.write("No recommendations available.")

# Initiate Home Page
# type 'streamlit run Home.py' in terminal to run the app
if __name__ == "__main__":
    Home.load_home_page()
