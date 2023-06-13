import datetime
import streamlit as st
import plotly.graph_objects as go

def next_week_range():
    next_week = datetime.date.today() + datetime.timedelta(weeks=1)
    desired_week_start = datetime.datetime.strptime(next_week.strftime("%Y-W%W") + '-1', "%Y-W%W-%w")
    desired_week_end = desired_week_start + datetime.timedelta(days=4)
    return desired_week_start, desired_week_end

def plot_title(start_date, end_date):
    start_week = start_date.strftime("%W")
    end_week = end_date.strftime("%W")
    if start_week == end_week:
        title_str = f"Week {start_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    else:
        title_str = f"Week {start_week}-{end_week}: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}"
    return title_str

def admin_app(model, data, headcount):
    week_start, week_end = next_week_range()
    dates = st.date_input("Choose dates", (week_start, week_end), max_value=week_end+datetime.timedelta(weeks=1))
    if len(dates) != 2:
        st.stop()
    start_date, end_date = dates
    true_data = data.loc[start_date:end_date]
    pred_data = true_data.drop(["actual"], axis=1)#.dropna()
    #true_data['predictions'] = model.predict(fh=np.arange(1,6), X=pred_data)
    true_data['predictions'] = model.predict(X=pred_data)
    
    copy_data = true_data.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=copy_data.index.strftime("%A %d/%m"), y=copy_data['actual'], name="Actual", line_color="#2ca02c"))
    if true_data['actual'].isna().sum() > 0:
        fig.add_trace(go.Scatter(x=true_data.index.strftime("%A %d/%m"), y=true_data['predictions'], name="Prediction", line_color="#1f77b4"))
    
    fig.update_layout(xaxis_title="Week day",
                    yaxis_title="Number of eating guests",
                    showlegend=True,
                    title_x = 0.5,
                    title_text=plot_title(start_date, end_date))
    fig.update_yaxes(range=[0,headcount])
   
    # Horizontal lines
    fig.add_scatter(x=[end_date, end_date], y=[0,90], name="Slow day", line=dict(color="Green"))
    fig.add_scatter(x=[end_date, end_date], y=[90,180], name="Business as usual", line=dict(color="Yellow"))
    fig.add_scatter(x=[end_date, end_date], y=[180,240], name="Heavy load", line=dict(color="Orange"))
    fig.add_scatter(x=[end_date, end_date], y=[240,302], name="Full house", line=dict(color="Red"))

    fig.add_hline(y=90, line_width=1, line_dash="dash", line_color="Blue")

    st.plotly_chart(fig, use_container_width=True)