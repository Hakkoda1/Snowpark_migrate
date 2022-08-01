# Snowflake packages
import snowflake.snowpark
from snowflake.snowpark.functions import sproc
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import *
from snowflake.snowpark.functions import udf
from PIL import Image


#Python packages
import sys
import cachetools
import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

#Snowflake connection info is saved in config.py
from config import snowflake_conn_prop
image_s = Image.open('logo_s.png')
image = Image.open('logo.png')
st.set_page_config(
     page_title="Hakkoda Housing",
     page_icon=image_s,
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://hakkoda.io/contact/',
         'About': "This is an example app powered by Snowpark for Python and Streamlit"
     }
)
# data=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
def create_session_object():    
    rolename = "SYSADMIN"
    dbname = "DEMO"
    schemaname = "TEST"
    warehouse = "streamlit"
    session = Session.builder.configs(snowflake_conn_prop).create()
    session.sql(f"USE ROLE {rolename}").collect()
    session.sql(f"USE WAREHOUSE {warehouse}").collect()
    session.sql(f"USE SCHEMA {dbname}.{schemaname}").collect()
    session.add_packages('snowflake-snowpark-python', 'scikit-learn', 'pandas', 'numpy', 'joblib', 'cachetools', 'xgboost')
    print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    session.add_import("@MODELS/housing_reg.joblib") 

    @cachetools.cached(cache={})
    def read_file(filename):
        import_dir = sys._xoptions.get("snowflake_import_directory")
        if import_dir:
                with open(os.path.join(import_dir, filename), 'rb') as file:
                        m = joblib.load(file)
                        return m
                    
    features = ['LONGITUDE', 'LATITUDE', 'HOUSING_MEDIAN_AGE', 'TOTAL_ROOMS',
        'TOTAL_BEDROOMS', 'POPULATION', 'HOUSEHOLDS', 'MEDIAN_INCOME', 'OCEAN_PROXIMITY']

    @udf(name='predict', is_permanent=True, stage_location='@udf', replace=True, session=session)
    def predict(LONGITUDE: float, LATITUDE: float, HOUSING_MEDIAN_AGE: float, TOTAL_ROOMS: float, 
                        TOTAL_BEDROOMS: float, POPULATION: float, HOUSEHOLDS: float, MEDIAN_INCOME: float, 
                        OCEAN_PROXIMITY: str) -> float:
        m = read_file('housing_reg.joblib')       
        row = pd.DataFrame([locals()], columns=features)
        return m.predict(row)[0]    
    return session

def load_data(session):
    #Load data
    housing_snowflake = session.table('housing')
    data = housing_snowflake.toPandas()
    
    #Model udf 
    

    # Add header and a subheader and position
    col1, col2, col3 = st.columns([3,3,3])
    with col1:
        st.write("")
    with col2:
        st.image(image, caption='Empowering data-driven organizations', width=400)
    with col3:
        st.write("")
        
    st.header("Housing Model Example")
    st.subheader("Powered by Snowpark for Python | Made with Streamlit")
    ###################################
        
    with st.container():
        st.subheader('Housing dataset')
        
        st.write(data)
       
######## Prediction options
    col_option1, col_option2, col_option3 = st.columns([3,3,3])
    with col_option1:
        LONGITUDE = st.selectbox('LONGITUDE',(data['LONGITUDE'].unique()))
        LATITUDE= st.selectbox('LATITUDE',(data['LATITUDE'].unique()))
        HOUSING_MEDIAN_AGE= st.selectbox('HOUSING_MEDIAN_AGE',(data['HOUSING_MEDIAN_AGE'].unique()))
    with col_option2:   
        TOTAL_ROOMS= st.selectbox('TOTAL_ROOMS',(data['TOTAL_ROOMS'].unique()))
        TOTAL_BEDROOMS= st.selectbox('TOTAL_BEDROOMS',(data['TOTAL_BEDROOMS'].unique()))
        POPULATION= st.selectbox('POPULATION',(data['POPULATION'].unique()))
    with col_option3:
        HOUSEHOLDS= st.selectbox('HOUSEHOLDS',(data['HOUSEHOLDS'].unique()))
        MEDIAN_INCOME= st.selectbox('MEDIAN_INCOME',(data['MEDIAN_INCOME'].unique()))
        OCEAN_PROXIMITY= st.selectbox('OCEAN_PROXIMITY',(data['OCEAN_PROXIMITY'].unique()))
    
    dataset_to_predict= pd.DataFrame(
        {'LONGITUDE': [LONGITUDE],
        'LATITUDE': [LATITUDE],
        'HOUSING_MEDIAN_AGE':[HOUSING_MEDIAN_AGE],
        'TOTAL_ROOMS':[TOTAL_ROOMS], 
        'TOTAL_BEDROOMS':[TOTAL_BEDROOMS] ,
        'POPULATION':[POPULATION] ,
        'HOUSEHOLDS':[HOUSEHOLDS] ,
        'MEDIAN_INCOME':[MEDIAN_INCOME] ,
        'OCEAN_PROXIMITY': [OCEAN_PROXIMITY] ,
        } 
    )
    
    st.write(dataset_to_predict)
 ############################  
 
 
 
 #prediction
    prediction=st.button("Make prediction")
    
    st.write(prediction)
    
    if prediction:
        to_predict=session.create_dataframe(dataset_to_predict)
        to_predict.write.mode("overwrite").save_as_table("Predict_one_row")
        
        
        
        
        snowdf_test = session.table("Predict_one_row")
        inputs = snowdf_test
        snowdf_results = snowdf_test.select(
                            *inputs,predict(*inputs).alias('HOUSE_VALUE_PREDICTION')
                            ).limit(20)  
        st.dataframe(snowdf_results.to_pandas())
        
if __name__ == "__main__":
    session = create_session_object()
    load_data(session)
    
    
#Run using streamlit run streamlit_app.py