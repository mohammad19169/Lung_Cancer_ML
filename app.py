import pickle
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing import image
from tempfile import NamedTemporaryFile
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(page_title='Lung Cancer Detection')

# Custom model loader to handle DepthwiseConv2D version mismatch
def load_custom_model(model_path):
    """Handles version mismatch for DepthwiseConv2D by filtering the 'groups' parameter"""
    with CustomObjectScope({
        'DepthwiseConv2D': lambda **kwargs: DepthwiseConv2D(**{k: v for k, v in kwargs.items() if k != 'groups'})
    }):
        return load_model(model_path, compile=False)

# Loading models with caching
@st.cache_resource
def load_cancer_model():
    return pickle.load(open('models/final_model.sav', 'rb'))

@st.cache_resource 
def load_cnn_model():
    try:
        return load_custom_model("models/keras_model.h5")
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        return None

cancer_model = load_cancer_model()
cnn_model = load_cnn_model()

# Sidebar navigation
with st.sidebar:
    selection = option_menu(
        'Lung Cancer Detection System',
        [
            'Introduction',
            'About the Dataset', 
            'Lung Cancer Prediction',
            'CNN Based Disease Prediction'
        ],
        icons=['activity', 'heart', 'person', 'image'],
        default_index=0
    )

# Introduction page
if selection == 'Introduction':
    st.image(Image.open("images/lung-cancer.jpg"), caption='Introduction to Lung Cancer', width=600)
    
    st.title('How common is lung cancer?')
    st.write("""
    Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States.
    In men, prostate cancer is more common, while in women breast cancer is more common.
    """)
    
    st.markdown("""
    The American Cancer Society's estimates for lung cancer in the US for 2023 are:
    - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)
    """)

    st.title("Is Smoking the only cause?")
    st.image(Image.open("images/menwa.png"), caption='Smoking is not the major cause', width=650)
    
    st.write("""
    The association between air pollution and lung cancer has been well established for decades.
    The International Agency for Research on Cancer classified outdoor air pollution as carcinogenic to humans in 2013.
    """)

    st.markdown("""
    - A 2012 study found that 52.1% of lung cancer patients had no history of smoking
    - 88% of female lung cancer patients were non-smokers
    - Only 41.8% of male patients were non-smokers
    """)

    st.title("Not just a Lahore phenomenon")
    st.image(Image.open("images/stove.png"), caption='Indoor pollution is also a major cause', width=650)
    
    st.markdown("""
    - 87.6% of adults in Pakistan had no smoking exposure
    - 97.9% of women in Pakistan had never smoked
    - Only 77.8% of men in Pakistan had never smoked
    """)

# Dataset page
if selection == 'About the Dataset':
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset analysis", "Training Data", "Test Data", "Algorithms Used", 'CNN Based Identification'
    ])

    with tab1:
        st.header("Lung Cancer Dataset")
        data = pd.read_csv("datasets/data.csv")
        st.write(data.head(10))
        
        st.code("""
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
        'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
        'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
        'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
        'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
        'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'],
        dtype='object')
        """, language='python')

        st.header("Pearson Correlation Matrix")
        st.image(Image.open("images/coors.png"), caption='Pearson Correlation Matrix', width=800)
        
        st.write("""
        Attributes with high correlation that could be dropped:
        """)
        
        st.code("""
        {'Chest Pain',
         'Coughing of Blood',
         'Dust Allergy', 
         'Genetic Risk',
         'OccuPational Hazards',
         'chronic Lung Disease'}
        """, language='python')

    with tab2:
        st.header("Lung Cancer Training Dataset")
        st.subheader("X_Train Data")
        data = pd.read_csv("datasets/train.csv", index_col=0)
        st.write(data)
        
        st.code("""
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
        'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
        'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
        'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')
        """, language='python')
        
        st.subheader("Y_Train Data")
        data = pd.read_csv("datasets/trainy.csv", index_col=0)
        st.dataframe(data, use_container_width=True)

    with tab3:
        st.header("Lung Cancer Test Dataset")
        st.subheader("X_Test Data")
        data = pd.read_csv("datasets/testx.csv", index_col=0)
        st.write(data)
        
        st.code("""
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
        'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
        'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
        'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')
        """, language='python')
        
        st.subheader("Y_Test Data")
        data = pd.read_csv("datasets/testy.csv", index_col=0)
        st.dataframe(data, use_container_width=True)
        
    with tab4:
        st.header("List of Algorithms Used")
        st.image(Image.open("images/algo.png"), caption='ML Algorithms', width=500)

        st.write("""
        Supervised Learning Algorithms used:
        - Linear Regression
        - Support Vector Machine  
        - K-Nearest Neighbours (KNN)
        - Decision Tree Classifier
        """)
        
        st.write("Model accuracies:")
        st.code("""
        The accuracy of the SVM is: 95 %
        The accuracy of the SVM is: 100 %
        The accuracy of Decision Tree is: 100 %
        The accuracy of KNN is: 100 %
        """, language='python')

        st.header("Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open("images/lg.png"), caption='LG Confusion Matrix', width=350)
        with col2:
            st.image(Image.open("images/svm.png"), caption='SVM Confusion Matrix', width=390)

    with tab5:
        st.header("Convolutional Neural Network Model")
        st.write("""
        [Chest CTScan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
        """)

        st.subheader("Approach:")
        st.markdown("""
        - Used Keras API with 2D Convolution and MaxPooling Layers
        - Sigmoid activation for binary classification
        - RMSprop optimizer with learning rate 0.001
        """)

        st.subheader("Model Summary")
        st.image(Image.open("images/summary.png"), caption='Model Summary', width=700)
        
        st.code("""
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        """, language='python')

        st.image(Image.open("images/epoc.png"), caption='Training Epochs', width=700)

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open("images/acc.png"), caption='Training vs Validation Accuracy', width=350)
        with col2:
            st.image(Image.open("images/loss.png"), caption='Training vs Validation Loss', width=350)

# Lung Cancer Prediction page
if selection == 'Lung Cancer Prediction':
    st.title('Lung Cancer Prediction using ML')

    # Load test data
    testx = pd.read_csv("datasets/testx.csv", index_col=0)
    testy = pd.read_csv("datasets/testy.csv", index_col=0)
    testx.reset_index(drop=True, inplace=True)
    testy.reset_index(drop=True, inplace=True)
    concate_data = pd.concat([testx, testy], axis=1)

    # Slider to select test case
    idn = st.slider('Select test case index', 0, len(concate_data)-1, 25)
    st.write(f"Displaying values of index {idn}")
    
    if st.button('Show this test case'):
        st.write(list(concate_data.iloc[idn]))

    # Get values from selected test case
    values = concate_data.iloc[idn]
    Age = values[0]
    Gender = values[1]
    AirPollution = values[2]
    Alcoholuse = values[3]
    BalancedDiet = values[4]
    Obesity = values[5]
    Smoking = values[6]
    PassiveSmoker = values[7]
    Fatigue = values[8]
    WeightLoss = values[9]
    ShortnessofBreath = values[10]
    Wheezing = values[11]
    SwallowingDifficulty = values[12]
    ClubbingofFingerNails = values[13]
    FrequentCold = values[14]
    DryCough = values[15]
    Snoring = values[16]

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input('Age', value=Age)
    with col2:
        Gender = st.text_input('Gender', value=Gender)
    with col3:
        AirPollution = st.text_input('Air Pollution', value=AirPollution)

    with col1:
        Alcoholuse = st.text_input('Alcohol Use', value=Alcoholuse)  
    with col2:
        BalancedDiet = st.text_input('Balanced Diet', value=BalancedDiet)
    with col3:
        Obesity = st.text_input('Obesity', value=Obesity)
        
    with col1:
        Smoking = st.text_input('Smoking', value=Smoking)
    with col2:
        PassiveSmoker = st.text_input('Passive Smoker', value=PassiveSmoker)
    with col3:
        Fatigue = st.text_input('Fatigue', value=Fatigue)
        
    with col1:
        WeightLoss = st.text_input('Weight Loss', value=WeightLoss)
    with col2:
        ShortnessofBreath = st.text_input('Shortness of Breath', value=ShortnessofBreath)
    with col3:
        Wheezing = st.text_input('Wheezing', value=Wheezing)
        
    with col1:
        SwallowingDifficulty = st.text_input('Swallowing Difficulty', value=SwallowingDifficulty)
    with col2:
        ClubbingofFingerNails = st.text_input('Clubbing of Finger Nails', value=ClubbingofFingerNails)
    with col3:
        FrequentCold = st.text_input('Frequent Cold', value=FrequentCold)
        
    with col1:
        DryCough = st.text_input('Dry Cough', value=DryCough)    
    with col2:
        Snoring = st.text_input('Snoring', value=Snoring)

    # Prediction button
    if st.button('Predict Lung Cancer Risk'):
        try:
            prediction = cancer_model.predict([[
                Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, 
                Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, 
                Wheezing, SwallowingDifficulty, ClubbingofFingerNails, 
                FrequentCold, DryCough, Snoring
            ]])
            
            if prediction[0] == 'High':
                st.error('High risk of lung cancer')
            elif prediction[0] == 'Medium':
                st.warning('Medium risk of lung cancer')
            else:
                st.success('Low risk of lung cancer')
                st.balloons()
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    expander = st.expander("Sample Test Data")
    expander.write(concate_data.head(5))

# CNN Based Prediction page
if selection == 'CNN Based Disease Prediction':
    st.title('Lung Cancer Detection using CNN and CT-Scan Images')
    
    if cnn_model is None:
        st.error("CNN model failed to load. Please check the model file.")
        st.stop()

    uploaded_file = st.file_uploader("Upload CT-Scan Image", type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        try:
            # Display file info
            st.write({
                "Filename": uploaded_file.name,
                "Type": uploaded_file.type,
                "Size": f"{uploaded_file.size / 1024:.2f} KB"
            })
            
            # Display image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded CT-Scan', use_container_width=True)
            
            # Preprocess image
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = cnn_model.predict(img_array)
            confidence = prediction[0][0]
            
            if confidence >= 0.5:
                st.balloons()
                st.success(f"Normal Case ({confidence:.2%} confidence)")
            else:
                st.error(f"Lung Cancer Case ({1-confidence:.2%} confidence)")
                
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")

# Hide Streamlit style elements
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)