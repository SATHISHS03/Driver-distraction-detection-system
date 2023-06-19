import base64
import streamlit as st
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
PAGE_TRANSITIONS = """
<style>
.page-enter {
    opacity: 0;
    transform: translateY(50px);
    transition: all 0.5s ease;
}
.page-enter-active {
    opacity: 1;
    transform: translateY(0);
    transition: all 0.5s ease;
}
.page-exit {
    opacity: 1;
    transform: translateY(0);
    transition: all 0.5s ease;
}
.page-exit-active {
    opacity: 0;
    transform: translateY(-50px);
    transition: all 0.5s ease;
}
sidebar .sidebar-content {
        background-color: #F5F5F5;
        font-family: 'Open Sans', sans-serif;
        padding: 25px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content .sidebar-section {
        margin-bottom: 25px;
    }
    .sidebar .sidebar-content .sidebar-section h2 {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li {
        list-style: none;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li a {
        color: #555555;
        text-decoration: none;
        display: flex;
        align-items: center;
    }
    .sidebar .sidebar-content .sidebar-section ul li a svg {
        margin-right: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li a:hover {
        color: #0072C6;
    }
</style>
"""
# Display CSS styles
st.markdown(PAGE_TRANSITIONS, unsafe_allow_html=True)
# Define page content
def home_page(image_file):    
    # st.subheader('welcome to the Med AI web! please select a domain form the sidebar')
    st.title("Driver distraction detection system")

    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
     # Set the background image of the body element
        # Set the background image of the body element
    st.markdown(
        f"""
        <style>
        .stApp  {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-size: 100% 90%;
        }}
        </style>
        """,
        
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader('Upload a Driving position image [ jpg , jpeg , png ]🔽', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
#Load the model
        test_model = load_model('vgg16_DDD.h5')

        activity_map = {'c0': 'Safe driving', 
                        'c1': 'Texting - right', 
                        'c2': 'Talking on the phone - right', 
                        'c3': 'Texting - left', 
                        'c4': 'Talking on the phone - left', 
                        'c5': 'Operating the radio', 
                        'c6': 'Drinking', 
                        'c7': 'Reaching behind', 
                        'c8': 'Hair and makeup', 
                        'c9': 'Talking to passenger'}

        # Load the image
        img = Image.open(uploaded_file)

        # Convert to RGB format
        img_rgb = img.convert('RGB')

        # Resize the image
        img_resized = img_rgb.resize((224, 224))

        # Convert to NumPy array
        img_array = np.array(img_resized)

        # Reshape to add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = test_model.predict(img_batch)

        # Get the predicted class label
        predicted_class_index = np.argmax(prediction)
        print(predicted_class_index)
        predicted_class_label = activity_map[f'c{predicted_class_index}']
        st.image(uploaded_file)

        st.warning(predicted_class_label)
            
def page_transition(next_page):
    st.markdown(f'<div class="page-exit page-exit-active">{st.session_state.current_page}</div>', unsafe_allow_html=True)
    st.session_state.current_page = next_page
    st.markdown(f'<div class="page-enter page-enter-active">{next_page}</div>', unsafe_allow_html=True)

# Initialize app
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'


home_page('istockphoto-996490224-612x612.jpg')