# Import Python packages
import streamlit as st
import json
import pydeck as pdk
import numpy as np
import pandas as pd

# Import Snowflake modules
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark import Window

# Get the current credentials
session = get_active_session()

# Add header and a subheader
st.header("Predicted Shift Sales by Location :earth_americas:")
st.subheader("Data-driven recommendations for food truck drivers :truck: :bento:")

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
    #pandas Data Frame for easier manipulation and visualisation in Streamlit
    return snowpark_df.to_pandas()


# Update predictions and plot when the "Update" button is clicked
if st.button("Update"):
    # Get predictions
    with st.spinner("Getting predictions..."):
        predictions = get_predictions(city, shift)

# Get the latitude and longitude of the selected city
    city_coords = predictions[["LATITUDE", "LONGITUDE"]].mean().values

    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=city_coords[0],
            longitude=city_coords[1],
            zoom=11,
            pitch=70,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=predictions,
                get_position=['LONGITUDE', 'LATITUDE'],
                radius=100,
                elevation_scale=2,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            
            )
        ]
    ))
        
