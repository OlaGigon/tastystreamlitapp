

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

# Update predictions and plot when the "Update" button is clicked
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


############# u wyświetlenie data frame i jej pobranie do csv 

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

    
################ to działa poniej  ale kropki w tym samym romiarze

# Update predictions and plot when the "Update" button is clicked
if st.button("Update"):
    # Get predictions
    with st.spinner("Getting predictions..."):
        predictions = get_predictions(city, shift)

    
    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    st.map(predictions,
        zoom=8)




########################
#dodatkowa mapa, która poprawnie wrzuca scatter plot na mapę 

    

# we will be working with this one particular event as an example
event_id_select = 10602642

# get all the data for this event
df_event = session.table('EVENTWATCH_AI.PUBLIC.EVENTWATCH_ALL_SAMPLE_V2')\
.filter(f"event_id = {event_id_select}").sort('event_publication_date')

# get the sites affected by the event
df_event_sites =df_event.select('supplier_name','site_name',
                            'site_worst_recovery_time_weeks', 'site_average_recovery_time_weeks',
                            'site_latitude','site_longitude', 'site_alternate_name') 


df_region=df_event.limit(1).select(F.col('EVENT_REGION_COORDINATES').alias('coordinates')).to_pandas()

event_row=df_event.first()


# parse the coordinates from the snowflake table with the JSON package
# and create a pandas DataFrame that Pydeck expects as data input
df_region_pandas = pd.DataFrame(data=json.loads(df_region["COORDINATES"].iloc[0]),columns=['COORDINATES'])

# this is how the polygon data looks like, it can have multiple rows
# if multiple regions are affected:
st.subheader("Polygon Data")
st.dataframe(df_region_pandas)


# create an array of polygons, in case we're dealing with multiple rows in the dataftame
my_polygons=[]

# Iterate over rows with polygons using iterrows()
for index, row in df_region_pandas.iterrows():
    coordinates_list = row["COORDINATES"]
    # Convert the list of coordinates into a numpy array
    coordinates_pairs = np.array(coordinates_list)

    # Calculate the center point so we can position the map
    center_point = np.average(coordinates_pairs, axis=0)


    polygon_layer_snow = pdk.Layer(
        "PolygonLayer",
        df_region_pandas.iloc[[index]],
        id="geojson",
        opacity=0.05,
        stroked=True,
        get_polygon="COORDINATES",
        filled=True,
        get_line_color=[200, 200, 200],
        auto_highlight=True,
        pickable=True,
    )

    my_polygons.append((polygon_layer_snow))

# create a HeatmapLayer with all the sites in the affected region
df_sites_geo =  df_event_sites.dropna().to_pandas()

sites_layer = pdk.Layer(
    'HeatmapLayer', 
    df_sites_geo,
    get_position=["SITE_LONGITUDE","SITE_LATITUDE"],
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1
)

r = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
        latitude=center_point[1],
        longitude=center_point[0],
        zoom=3,
        pitch=50,
    ),
    layers=[my_polygons,sites_layer]    
)

st.subheader("Affected Region and Factory Sites")
st.pydeck_chart(r)
