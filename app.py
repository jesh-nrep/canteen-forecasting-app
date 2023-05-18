import streamlit as st
import os
import pandas as pd
from pycaret.regression import load_model
from azure.storage.blob import BlobClient

from admin import admin_app
from user import user_app

def load_data():
    blob = BlobClient.from_connection_string(conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"], 
                                             container_name="data", 
                                             blob_name="processed_data.csv")
    blob_data = blob.download_blob()
    data = pd.read_csv(blob_data, parse_dates=['date'], index_col="date")
    data.rename({"total": "actual"}, axis=1, inplace=True)
    return data

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = 'user'
            del st.session_state["password"]  # don't store password
        elif st.session_state['password'] == st.secrets['admin_password']:
            st.session_state['password_correct'] = 'admin'
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return st.session_state['password_correct']

def main():
    st.title("Urban Partners Canteen Forecasting")
    model = load_model("regression_model", "azure", {"container": "models"})
    data = load_data()
    cond = check_password()
    if cond == "user":
        user_app(model, data)
    elif cond == "admin":
        admin_app(model, data)
        

if __name__ == "__main__":
    main()