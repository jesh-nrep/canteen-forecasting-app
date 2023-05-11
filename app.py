from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import datetime
import os
import numpy as np
import pandas as pd
from pycaret.regression import load_model
from azure.storage.blob import BlobClient

IMG_BINS = np.array([0.6, 0.8, 1.2, 1.4])
IMG_PATH = np.array([1, 2, 3, 4, 5])
IMG_LABELS = np.array(["Slow day", "Relaxed", "Somewhat busy", "Heavy load", "Full house"])

def next_week_range():
    next_week = datetime.date.today() + datetime.timedelta(weeks=1)
    desired_week_start = datetime.datetime.strptime(next_week.strftime("%Y-W%W") + '-1', "%Y-W%W-%w")
    desired_week_end = desired_week_start + datetime.timedelta(days=4)
    return desired_week_start, desired_week_end

def load_data():
    blob = BlobClient.from_connection_string(conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"], 
                                             container_name="data", 
                                             blob_name="processed_data.csv")
    blob_data = blob.download_blob()
    data = pd.read_csv(blob_data, parse_dates=['date'], index_col="date")
    data.rename({"total": "actual"}, axis=1, inplace=True)
    return data

def plot_title(start_date, end_date):
    start_week = start_date.strftime("%W")
    end_week = end_date.strftime("%W")
    if start_week == end_week:
        title_str = f"Week {start_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    else:
        title_str = f"Week {start_week}-{end_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    return title_str

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
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
        return True

def load_images(arr):
    imgs = []
    captions = []
    for el in arr:
        imgs.append(Image.open("imgs\\chef" + str(el) + ".png"))
        captions.append(IMG_LABELS[el-1])
    return imgs, captions

def main():
    if check_password():
        st.title("Urban Partners Canteen Forecasting")
        week_start, week_end = next_week_range()
        model = load_model("regression_model", "azure", {"container": "models"})
        data = load_data()
        image_binning = IMG_BINS*data['actual'].mean()
        image_binning = np.insert(image_binning, 0, data['actual'].min())
        image_binning = np.append(image_binning, data['actual'].max())
        start_date, end_date = st.date_input("Choose dates", (week_start, week_end), max_value=week_end+datetime.timedelta(weeks=1))
        true_data = data.loc[start_date:end_date]
        pred_data = true_data.drop(["actual"], axis=1)
        true_data['predictions'] = model.predict(X=pred_data)
        true_data['img_index'] = pd.cut(true_data['predictions'], bins=image_binning, labels=IMG_PATH)
        imgs, captions = load_images(true_data['img_index'])
        st.image(imgs, captions)

if __name__ == "__main__":
    main()