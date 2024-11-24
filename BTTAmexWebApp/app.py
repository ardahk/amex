import streamlit as st
import json
from RandomUserProfile import RandomUserProfile
from load_data import *

st.set_page_config(
    page_title="BTT Amex Product Recommendation System", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# Hide Streamlit anchors and full screen icons
st.markdown(
    "<style>[data-testid='stHeaderActionElements'] {display: none;} "
    "[data-testid='StyledFullScreenButton'] {display: none;}</style>",
    unsafe_allow_html=True,
)

class App:    
    # Load Home Page
    @staticmethod
    def load_home_page():
        st.subheader("BTT AI Studio and Amex Team #1 Presents:")
        st.header("User Product Recommendation System")
        st.write("This Application Demonstrates our Two-Tower Product Recommendation Model.")
        st.write("---")
  
        profiles = []  # List to store user profiles
        
        # Check if profiles already exist in a JSON file
        try:
            with open('user_profiles.json', 'r') as json_file:
                profiles = json.load(json_file)
        except FileNotFoundError:
            st.write("No existing profiles found. Generating new ones...")
            for i in range(6):
                user_id = get_random_user_id()
                user_profile = RandomUserProfile(user_id)
                profiles.append(user_profile.to_dict())
            
            # Save the profiles to a JSON file
            with open('user_profiles.json', 'w') as json_file:
                json.dump(profiles, json_file, indent=10)
        
        # Display profiles
        cols = st.columns(3)  
        for i, profile in enumerate(profiles):
            col = cols[i % 3]
            with col:
                user_link = "user_page?user_id=" + str(profile['user_id'])
                
                # Display user name and link
                #st.markdown(f"### [{profile['name']}]({user_link})")
                # Link to another Streamlit page in the same tab
                st.markdown(
                    f'<a href="{user_link}" target="_self" style="text-decoration:none;"><h3>{profile["name"]}</h3></a>',
                    unsafe_allow_html=True,
                )

                
                # Display user image (replace with your method to load an image)
                st.image(profile['image'])
                
            if (i + 1) % 3 == 0 and i < 5:
                st.write("---")
        
        
# Initiate App
if __name__ == "__main__":
    App.load_home_page()
