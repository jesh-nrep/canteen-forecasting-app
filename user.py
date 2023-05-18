import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

IMG_BINS = np.array([0.6, 0.8, 1.2, 1.4])
IMG_PATH = np.array([1, 2, 3, 4, 5])
IMG_LABELS = np.array(["Slow day", "Relaxed", "Somewhat busy", "Heavy load", "Full house"])

def get_dates_of_week(week_str: str):
    week = datetime.date.today()
    if week_str == "Last week":
        week -= datetime.timedelta(weeks=1)
    elif week_str == "Next week":
        week += datetime.timedelta(weeks=1)
    elif week_str == "In two weeks":
        week += datetime.timedelta(weeks=2)
    desired_week_start = datetime.datetime.strptime(week.strftime("%Y-W%W") + '-1', "%Y-W%W-%w")
    desired_week_end = desired_week_start + datetime.timedelta(days=4)
    return desired_week_start, desired_week_end

def load_images(arr):
    imgs = []
    captions = []
    for el in arr:
        imgs.append(Image.open("imgs/chef" + str(el) + ".png"))
        captions.append(IMG_LABELS[el-1])
    return imgs, captions

def user_app(model, data):
    image_binning = IMG_BINS*data['actual'].mean()
    image_binning = np.insert(image_binning, 0, data['actual'].min())
    image_binning = np.append(image_binning, data['actual'].max())
    option = st.selectbox('Whick week would you like to view?', options=('Last week', 'This week', 'Next week', "In two weeks"), index=1)
    start_date, end_date = get_dates_of_week(option)
    st.text(f"You have selected: {start_date.strftime('%d/%m')}-{end_date.strftime('%d/%m')}")
    true_data = data.loc[start_date:end_date]
    pred_data = true_data.drop(["actual"], axis=1)
    true_data['predictions'] = model.predict(X=pred_data)
    true_data['img_index'] = pd.cut(true_data['predictions'], bins=image_binning, labels=IMG_PATH)
    imgs, captions = load_images(true_data['img_index'])
    cols = st.columns(len(true_data))
    for i, col in enumerate(cols):
        with col:
            date_column = start_date + datetime.timedelta(days=i)
            st.write(date_column.strftime("%A %d/%m"))
            st.image(imgs[i], captions[i])