# Snowflake packages
import snowflake.snowpark
from snowflake.snowpark.functions import sproc
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import *
from snowflake.snowpark.functions import udf

#Python packages
import sys
import cachetools
import os
import pandas as pd
import numpy as np
import streamlit as st
import io
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

#Snowflake connection info is saved in config.py
from config import snowflake_conn_prop

st.set_page_config(
     page_title="Hakkoda Housing",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://hakkoda.io/contact/',
         'About': "This is an example app powered by Snowpark for Python and Streamlit"
     }
)

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
    return session
def load_data(session):
    housing_snowflake = session.table('housing')
    data = housing_snowflake.toPandas()
    # Add header and a subheader
    st.header("Hakkoda: Housing Model Example")
    st.subheader("Powered by Snowpark for Python | Made with Streamlit")

    
    with st.container():
        st.subheader('Housing Sample')
        st.dataframe(data.head())
   
if __name__ == "__main__":
    session = create_session_object()
    load_data(session)
    
#Run using streamlit run streamlit_app.py