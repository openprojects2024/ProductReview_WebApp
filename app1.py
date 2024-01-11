from tensorflow.keras.models import load_model 
from preprocessing import preprocess_reviews, image_path
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
            caption = os.path.basename(img_path).split('.')[0]
            with col_elem:
                #st.write("Example images with formatted captions:")
                st.markdown(f"<span style='text-align: center; color: yellow;font-size: 30px'>{caption}</span>", unsafe_allow_html=True)
                
def sentiment(result):
    if result<0.35:
        return 'Negative'
    elif result>=0.35 and result<0.65:
        return 'Neutral'
    else:
        return 'Positive'
Review_customer = {"Product_Name":"","Review":"","Sentiment":""}
review_data = pd.read_csv("Review_customer.csv")
review_data = review_data[["Product_Name","Review","Sentiment"]]

def write_review(selected_product,review_data):
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
    img_path = image_path(selected_product)
    print(img_path)
    imagee = cv2.imread(img_path)
    #print(imagee)
    imagee = cv2.resize(imagee,[640,480])
    cv2.imshow('Image', imagee)
    st.image(imagee, use_column_width=True)
    
    product_description(selected_product)
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
        review_data = review_data[["Product_Name","Review","Sentiment"]]
        review_data.to_csv("Review_customer.csv")

def product_description(selected_product):
    if selected_product == "American Tourist Bag":
        #st.title("Red Graphic Backpack")
        st.markdown("<h2 style='text-align: center; color: yellow;font-size: 30px'>Red Graphic American Tourist Backpack</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: yellow;font-size: 30px'>Price: Rs. 900</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**: 
            Red graphic backpack  
            Non-Padded haul loop  
            2 main compartments with zip closure  
            Padded Mesh back  
            Padded shoulder strap: Padded  
            Water-resistance: No  

            ---  
            **Size & Fit**  
            Height: 48 cm  
            Width: 33 cm  
            Depth: 29 cm  

            ---  
            **Material & Care**  
            Polyester  
            Wipe with a clean, dry cloth to remove dust  

            ---  
            **Specifications**  
            Back: Padded Mesh  
            Compartment Closure: Zip  
            Haul Loop Type: Non-Padded  
            Material: Polyester  
            Number of Main Compartments: 2  
            Number of Zips: 1  
            Occasion: Casual  
            Padded Shoulder Strap: Padded  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Iphone 15 Pro":
        #st.title("")
        st.markdown("<h2 style='text-align: center; color: yellow;font-size: 30px'>IPhone 15 Pro</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: yellow;font-size: 30px'>Price: Rs. 125900</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**
            iPhone 15 Pro  
            Super Retina XDR display  
            A15 Bionic chip  
            Triple-camera system with Ultra Wide, Wide, and Telephoto  
            Face ID  
            Water and dust resistance  

            ---  
            **Size & Weight**  
            Height: 146.7 mm  
            Width: 71.5 mm  
            Depth: 7.7 mm  
            Weight: 187 grams  

            ---  
            **Features**  
            Super Retina XDR display  
            Ceramic Shield front cover  
            Pro camera system  
            LiDAR Scanner  
            Night mode  
            5G capable  

            ---  
            **Battery Life**  
            Up to 75 hours audio playback  
            Up to 19 hours talk time (5G)  
            Up to 75 hours audio playback  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Refrigerator (Samsung)":
        #st.title("Samsung Refrigerator")
        st.markdown("<h2 style='text-align: center; color: yellow;font-size: 30px'>Samsung Refrigerator</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: yellow;font-size: 30px'>Price: Rs. 22590</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**
            Samsung Refrigerator  
            Model: XYZ1234  
            Capacity: 600 liters  
            Energy Efficiency: A++  
            Frost-free  

            ---  
            **Key Features**  
            Twin Cooling Plus System  
            All-around Cooling  
            Digital Inverter Technology  
            Power Cool and Power Freeze  
            LED Lighting  

            ---  
            **Dimensions**  
            Height: 1780 mm  
            Width: 910 mm  
            Depth: 716 mm  

            ---  
            **Additional Information**  
            External Water Dispenser  
            Multi-ventilation  
            Smart Conversion  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True


                   

def main(selection):
    #st.sidebar.title("Sidebar Menu")
    #selection = st.sidebar.radio("Go to", ["Home", "Product List", "Contact", "Settings"])

    if selection == "Home":
        home()
    elif selection == "Products":
        product_list()
    elif selection == "Write Review":
        st.markdown("<h1 style='text-align: center; color: yellow;'>Give Your Valuable Feedback</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color:white;font-size: 20px'>Select a product to give your review</span>", unsafe_allow_html=True)
        selected_product = st.selectbox("",image_paths_individual['Product_Name'].unique())
        write_review(selected_product,review_data)
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
    folder_path = "C:/Users/krishnendu/VSCODE_projects/ML-recruitment-assessment/products/products"
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




            
            
            
            








            

