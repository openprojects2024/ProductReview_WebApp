from tensorflow.keras.models import load_model 
from preprocessing import preprocess_reviews
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import os
import cv2
import pandas as pd

model = load_model('models/LSTM.h5')
token = pickle.load((open('token.pkl', 'rb')))
max_len = 130
st.set_page_config (layout="wide")
image_paths_individual = pd.read_csv("image_path.csv")


def find_star(result):
    if result >= 0.8:
        st.markdown("<span style='color:green;font-size: 80px'>* * * * *</span>", unsafe_allow_html=True)
    elif result>0.65:
        st.markdown("<span style='color:grreen;font-size: 80px'>* * * *</span>", unsafe_allow_html=True)
    elif result>0.5:
        st.markdown("<span style='color:white;font-size: 80px'>* * *</span>", unsafe_allow_html=True)
    elif result>0.3:
        st.markdown("<span style='color:red;font-size: 80px'>* *</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;font-size: 80px'>*</span>", unsafe_allow_html=True)

def display_images_from_folder(folder_path):
    image_extensions = ["png", "jpg", "jpeg", "gif"]  # Add other extensions if needed

    image_files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if any(filename.lower().endswith(ext) for ext in image_extensions)
    ]
    num_images = len(image_files)
    num_columns = 2 # Change the number of columns as needed

    for i in range(0, num_images, num_columns):
        images_row = image_files[i:i + num_columns]
        col1,col2 = st.columns([100,100])
        for img_path, col_elem in zip(images_row, [col1,col2]):
            imagee = cv2.imread(img_path)
            imagee = cv2.resize(imagee,[640,480])
            cv2.imshow('Image', imagee)
            col_elem.image(imagee, use_column_width=True)
            caption = os.path.basename(img_path)
            with col_elem:
                #st.write("Example images with formatted captions:")
                st.markdown("<span style='text-align: center; color: yellow;font-size: 30px'>IPhone 15 Pro:</span>", unsafe_allow_html=True)
                
def sentiment(result):
    if result<0.35:
        return 'Negative'
    elif result>=0.35 and result<0.65:
        return 'Neutral'
    else:
        return 'Positive'
Review_customer = {"Product_Name":"","Review":"","Sentiment":""}
review_data = pd.read_csv("Review_customer.csv")

def write_review(selected_product):
    ## Back ground Image entry
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_element, unsafe_allow_html=True)
    
    ## This will be chage according to the different product images and can be set more interactive.
    ## this will take some more time, so as of now I am setting it for the singe image
    img_path = "C:/Users/krishnendu/VSCODE_projects/ML-recruitment-assessment/products/iphone15_pro/1.jpg"
    imagee = cv2.imread(img_path)
    imagee = cv2.resize(imagee,[640,480])
    cv2.imshow('Image', imagee)
    st.image(imagee, use_column_width=True)
    
    ## Take the review form the user for the selected product
    st.markdown("<span style='color:white;font-size: 20px'>Enter the review</span>", unsafe_allow_html=True)
    #st.text_area("This is the text area")
    review = st.text_area("Enter your review",label_visibility = 'collapsed')
    review = str(review)
    if st.button("Find Rating "):
        review_encoded = preprocess_reviews(review, max_len,token)
        result = model.predict(review_encoded)
        find_star(result)
        Review_customer["Sentiment"] = (str(sentiment(result)))
        Review_customer["Product_Name"] = (str(selected_product))
        Review_customer["Review"] = (str(review))
        review_data.loc[review_data.shape[0]] = Review_customer
        review_data.to_csv("Review_customer.csv")
               

def main(selection):
    #st.sidebar.title("Sidebar Menu")
    #selection = st.sidebar.radio("Go to", ["Home", "Product List", "Contact", "Settings"])

    if selection == "Home":
        home()
    elif selection == "Products":
        product_list()
    elif selection == "Write Review":
        st.markdown("<h1 style='text-align: center; color: yellow;'>Give Your Valuable Feedback</h1>", unsafe_allow_html=True)
        product_review = pd.read_csv("product_review.csv")
        st.markdown("<span style='color:white;font-size: 20px'>Select a product to give your review</span>", unsafe_allow_html=True)
        selected_product = st.selectbox("",product_review['Product_Name'].unique())
        write_review(selected_product)
        st.dataframe(review_data)
    elif selection == "Settings":
        settings()

def product_list():
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_element, unsafe_allow_html=True)
    st.title("")
    st.markdown("<h1 style='text-align: center; color: yellow;'>Available Products</h1>", unsafe_allow_html=True)
    folder_path = "C:/Users/krishnendu/VSCODE_projects/ML-recruitment-assessment/products/iphone15_pro"
    display_images_from_folder(folder_path)
    
    
    
def home():
    st.title("")
    st.markdown("<h1 style='text-align: center; color: yellow;'>Product Review Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<span style='text-align: center; color: white;'>Product : Go to the Product section to see the products available</span>", unsafe_allow_html=True)
    st.markdown("<span style='text-align: center; color: white;'>Write Review : Go to the Review section to put your review about the products</span>", unsafe_allow_html=True)
    
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
    background-size: cover;
    }
    </style>
    """

    st.markdown(page_element, unsafe_allow_html=True)
    
    



selected = option_menu(
    menu_title = None,
    options = ["Home", "Products","Write Review"],
    icons = ["house",'bag-dash','star-half'],
    menu_icon = 'cast',
    orientation = 'horizontal'
)
main(selected)




            
            
            
            








            

