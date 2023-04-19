import streamlit as st
import datetime

def week_range_today():
    today = datetime.date.today()
    desired_week_start = datetime.datetime.strptime(today.strftime("%Y-W%W") + '-1', "%Y-W%W-%w")
    desired_week_end = desired_week_start + datetime.timedelta(days=4)
    return desired_week_start, desired_week_end

def main():
    st.title("Canteen Forecasting")
    week_start, week_end = week_range_today()
    d = st.date_input("Choose dates", (week_start, week_end), max_value=week_end+datetime.timedelta(weeks=1))
    st.write("Forecast for:", " - ".join(i.strftime("%Y-%m-%d") for i in d))

if __name__ == "__main__":
    main()