import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

IMG_BINS = np.array([0.3, 0.6, 0.8])
IMG_PATH = np.array([1, 2, 3, 4])
IMG_LABELS = np.array(["Slow day \n (1/4)", "Business as usual \n (2/4)", "Heavy load \n (3/4)", "Full house \n (4/4)"])
rounding_func = lambda x: round(x/10)*10 # Round to nearest 10

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
        imgs.append(Image.open("imgs/chef" + str(el) + ".jpg"))
        captions.append(IMG_LABELS[el-1])
    return imgs, captions

def explainer_load_images():
    imgs = []
    captions = []
    for i, path in enumerate(IMG_PATH):
        imgs.append(Image.open("imgs/chef" + str(path) + ".jpg"))
        captions.append(IMG_LABELS[i])
    return imgs, captions


def user_app(model, data, headcount):
    image_binning = IMG_BINS*headcount
    image_binning = np.insert(image_binning, 0, 0)
    image_binning = np.append(image_binning, headcount)
    option = st.selectbox('Whick week would you like to view?', options=('Last week', 'This week', 'Next week', "In two weeks"), index=1)
    start_date, end_date = get_dates_of_week(option)
    st.write(f"You have selected: {start_date.strftime('%d/%m')}-{end_date.strftime('%d/%m')}")
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

    with st.expander("See explanation"):
        explainer_columns = st.columns(len(IMG_LABELS))
        explainer_imgs, explainer_captions = explainer_load_images()
        for i, col in enumerate(explainer_columns):
            with col:
                st.image(explainer_imgs[i], explainer_captions[i])
        st.write(f"""
        The categories are defined according to the head count in the Danish office. As of today, there are {headcount} employees in the Danish office.\n
        A slow day indicates that there will be less people in the office than normal (less than {rounding_func(image_binning[1])} people). Slow days often occurs on public holidays.\n
        Business as usual means that it is considered a normal day at the office (between {rounding_func(image_binning[1])}-{rounding_func(image_binning[2])} people).\n
        Heavy load is busier than usual (between {rounding_func(image_binning[2])}-{rounding_func(image_binning[3])} people).\n
        Full house is above {rounding_func(image_binning[3])} people and is the most busy.
        """)