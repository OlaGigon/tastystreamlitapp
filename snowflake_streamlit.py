

# Import Python packages
import plotly.express as px
import streamlit as st
import json

# Import Snowflake modules
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark import Window

# Get the current credentials
session = get_active_session()

# Set Streamlit page config
st.set_page_config(
    page_title="Streamlit App: Snowpark 101",
    page_icon=":truck:",
    layout="wide",
)

# Add header and a subheader
st.header("Predicted Shift Sales by Location")
st.subheader("Data-driven recommendations for food truck drivers.")


# Create input widgets for cities and shift
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # Drop down to select city
        city = st.selectbox(
            "City:",
            session.table("frostbyte_tasty_bytes_dev.analytics.shift_sales_v")
            .select("city")
            .distinct()
            .sort("city"),
        )

    with col2:
        # Select AM/PM Shift
        shift = st.radio("Shift:", ("AM", "PM"), horizontal=True)


# Get predictions for city and shift time
def get_predictions(city, shift):
    # Get data and filter by city and shift
    snowpark_df = session.table(
        "frostbyte_tasty_bytes_dev.analytics.shift_sales_v"
    ).filter((F.col("shift") == shift) & (F.col("city") == city))

    # Get rolling average
    window_by_location_all_days = (
        Window.partition_by("location_id")
        .order_by("date")
        .rows_between(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW - 1)
    )

    snowpark_df = snowpark_df.with_column(
        "avg_location_shift_sales",
        F.avg("shift_sales").over(window_by_location_all_days),
    ).cache_result()

    # Get tomorrow's date
    date_tomorrow = (
        snowpark_df.filter(F.col("shift_sales").is_null())
        .select(F.min("date"))
        .collect()[0][0]
    )

    # Filter to tomorrow's date
    snowpark_df = snowpark_df.filter(F.col("date") == date_tomorrow)

    # Impute
    snowpark_df = snowpark_df.fillna(value=0, subset=["avg_location_shift_sales"])

    # Encode
    snowpark_df = snowpark_df.with_column("shift", F.iff(F.col("shift") == "AM", 1, 0))

    # Define feature columns
    feature_cols = [
        "MONTH",
        "DAY_OF_WEEK",
        "LATITUDE",
        "LONGITUDE",
        "CITY_POPULATION",
        "AVG_LOCATION_SHIFT_SALES",
        "SHIFT",
    ]

    # Call the inference user-defined function
    snowpark_df = snowpark_df.select(
        "location_id",
        "latitude",
        "longitude",
        "avg_location_shift_sales",
        F.call_udf(
            "udf_linreg_predict_location_sales", [F.col(c) for c in feature_cols]
        ).alias("predicted_shift_sales"),
    )

    return snowpark_df.to_pandas()

#px.set_mapbox_access_token()

# Update predictions and plot when the "Update" button is clicked - nie działa  mapa ;/ 
if st.button("Update"):
    # Get predictions
    with st.spinner("Getting predictions..."):
        predictions = get_predictions(city, shift)

    
    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    fig = px.scatter_mapbox(
    #px.density_mapbox(
        predictions,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="LOCATION_ID",
        size="PREDICTED_SHIFT_SALES",
        color="PREDICTED_SHIFT_SALES",
        zoom=8,
        height=800,
        width=1000,
    )
    fig.update_layout(mapbox_style="open-street-map")
    #fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)


############# tu wyświetlenie data frame i jej pobranie do csv 

# Update predictions and plot when the "Update" button is clicked
if st.button("Update"):
    # Get predictions
    with st.spinner("Getting predictions..."):
        predictions = get_predictions(city, shift)

    
    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    fig = px.scatter_mapbox(
        predictions,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="LOCATION_ID",
        size="PREDICTED_SHIFT_SALES",
        color="PREDICTED_SHIFT_SALES",
        zoom=8,
        height=800,
        width=1000,
    )
    fig.update_layout(mapbox_style="open-street-map")
    #fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    st.write(pd.DataFrame(predictions))
    csv = convert_df(pd.DataFrame(predictions))
    st.download_button(
    "Press to Download",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

    
################ to działa poniej  ale bez wymiaru punktów

# Update predictions and plot when the "Update" button is clicked
if st.button("Update"):
    # Get predictions
    with st.spinner("Getting predictions..."):
        predictions = get_predictions(city, shift)

    
    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    st.map(predictions,
        zoom=8)

