import streamlit as st
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import numpy as np
#from pycaret.time_series import load_model
from pycaret.regression import load_model
from azure.storage.blob import BlobClient

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
        title_str = f"Uge {start_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    else:
        title_str = f"Uge {start_week}-{end_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    return title_str

def main():
    st.title("NREP Kantine Forecasting")
    week_start, week_end = next_week_range()
    model = load_model("regression_model", "azure", {"container": "models"})
    data = load_data()
    start_date, end_date = st.date_input("Vælg datoer", (week_start, week_end), max_value=week_end+datetime.timedelta(weeks=1))
    true_data = data.loc[start_date:end_date]
    pred_data = true_data.drop(["actual"], axis=1)#.dropna()
    #true_data['predictions'] = model.predict(fh=np.arange(1,6), X=pred_data)
    true_data['predictions'] = model.predict(X=pred_data)
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)

    copy_data = true_data.dropna()
    ax.plot(copy_data.index.strftime("%A"), copy_data['actual'], color="green", label="Actuals")

    ax.plot(true_data.index.strftime("%A"), true_data['predictions'], color="blue", label="Predictions")
    ax.legend()
    ax.set_title(plot_title(start_date, end_date))
    ax.set_ylim(0, 300)
    st.pyplot(fig)


if __name__ == "__main__":
    main()