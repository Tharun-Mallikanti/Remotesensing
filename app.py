from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64
import seaborn as sns
import datacube
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import odc.algo
import csv
import plotly.graph_objs as go
import plotly.offline as pyoff
import plotly.io as pio
import xarray as xr
from geopy.geocoders import Nominatim
matplotlib.use('Agg')
def get_area_name(latitude, longitude):
    geolocator = Nominatim(user_agent='my-app')  # Initialize the geocoder
    location = geolocator.reverse((latitude, longitude))  # Reverse geocode the coordinates
    if location is not None:
        address_components = location.raw['address']
        city_name = address_components.get('city', '')
        if not city_name:
            city_name = address_components.get('town', '')
        if not city_name:
            city_name = address_components.get('village', '')
        return city_name
    else:
        return "City name not found"

dc = datacube.Datacube(app="Flask_Text")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/analyse', methods=['GET'])
def analyse():
    return render_template("analyse.html")

@app.route('/type/<analysis_type>', methods=['GET','POST'])
def analysis(analysis_type):
    if request.method=="POST":
        data = request.get_json()
        print(data)
        try:
            dc = datacube.Datacube(app='water_change_analysis')
            coordinates = data['coordinates']
            time_range = (data['fromdate'], data['todate'])
            study_area_lat = (coordinates[0][0], coordinates[1][0])
            study_area_lon = (coordinates[1][1], coordinates[2][1])
            ds = dc.load(product=['s2a_sen2cor_granule',"s2b_sen2cor_granule"],
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:6933',
                resolution=(-30, 30),
                dask_chunks={'time': 1}
            )
            ds = odc.algo.to_f32(ds)
            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            elif analysis_type=="evi":
                res=2.5*((ds.nir-ds.red)/(ds.nir+6*ds.red-7.5*ds.blue+1))
                print(res)
                res = xr.where(~np.isfinite(res), 0.0, res)
            elif analysis_type=="graph":
                ndvi=(ds.nir - ds.red) / (ds.nir + ds.red)
                evi=2.5*((ds.nir-ds.red)/(ds.nir+6*ds.red-7.5*ds.blue+1))
                evi = xr.where(~np.isfinite(evi), 0.0, evi)
                ndvi_threshold = 0.4
                evi_threshold = 0.2

                forest_mask_ndvi = np.where(ndvi > ndvi_threshold, 1, 0)
                forest_mask_evi = np.where(evi > evi_threshold, 1, 0)

                forest = np.logical_and(forest_mask_ndvi, forest_mask_evi)

                # Create forest masks based on NDVI and EVI thresholds
                dense_forest_mask = np.where((ndvi > ndvi_threshold) & (evi > evi_threshold), 1, 0)
                open_forest_mask = np.where((ndvi > ndvi_threshold) & (evi <= evi_threshold), 1, 0)
                sparse_forest_mask = np.where((ndvi <= ndvi_threshold) & (evi <= evi_threshold), 1, 0)

                # Calculate the area of each pixel
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])

                data = [['day', 'month', 'year', 'dense_forest', 'open_forest', 'sparse_forest', 'forest', 'total']]

                for i in range(dense_forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    new_data_time = data_time.split("-")
                    
                    # Calculate the forest cover area for each forest type
                    dense_forest_cover_area = np.sum(dense_forest_mask[i]) * pixel_area
                    open_forest_cover_area = np.sum(open_forest_mask[i]) * pixel_area
                    sparse_forest_cover_area = np.sum(sparse_forest_mask[i]) * pixel_area

                    # Calculate the total forest cover area
                    total_forest_cover_area = dense_forest_cover_area + open_forest_cover_area + sparse_forest_cover_area

                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area
                    
                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                dense_forest_cover_area, open_forest_cover_area,
                                sparse_forest_cover_area, total_forest_cover_area, original])
                    column_names = data[0]
                    df = pd.DataFrame(data[1:], columns=column_names)
                    df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                    print(df["year-month"])
                    X = df[["year", "month"]]
                    y = df["dense_forest"]
                    train_size = int(0.8 * len(df))
                    X_test = df[['year', 'month']].iloc[train_size:]
                    y_test = df['dense_forest'].iloc[train_size:]
                    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                    rf_regressor.fit(X, y)
                    y_pred = rf_regressor.predict(X)

                    df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                    X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')
                    X = df[["year", "month"]]
                y = df["forest"]

                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict([[2023, 5]])
                print(df, y_pred)

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')

                print("year-month done")

                plot_data = [
                    go.Scatter(
                        x = df['year-month'],
                        y = df['forest']/1000000,
                        name = "Forest Actual"
                    ),
                    go.Scatter(
                        x = ['2024-05'],
                        y = y_pred/1000000,
                        name = "Forest Predicted"
                    )
                ]

                print("Plot plotted")

                plot_layout = go.Layout(
                    title='Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)
                plot_json = pio.to_json(fig)

                area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
                print(area_name)

                return jsonify({"plot": plot_json, "type": "Random Forest Analysis", "area_name": area_name})
                    
            else:
                return jsonify({"error": "Invalid type"})
            
            if analysis_type=="ndvi":
                title = 'Vegetation'
                cmap = 'YlGn_r'
            elif analysis_type=="ndwi":
                title = 'Water'
                cmap = 'cividis'
            elif analysis_type=="evi":
                title = 'Enhanced vegetation index'
                cmap = 'viridis'

            sub_res = res.isel(time=[0, -1])

            mean_res = res.mean(dim=['x', 'y'], skipna=True)
            mean_res_rounded = np.array(list(map(lambda x: round(x, 4), mean_res.values.tolist())))
            
            mean_res_rounded = mean_res_rounded[np.logical_not(np.isnan(mean_res_rounded))]
            mean_res_rounded = [0 if (i>1 or i<-1) else i for i in mean_res_rounded]
            labels = list(map(lambda x: x.split('T')[0], [i for i in np.datetime_as_string(res.time.values).tolist()]))    

            plot = sub_res.plot(col='time', col_wrap=2)
            for ax, time in zip(plot.axes.flat, sub_res.time.values):
                ax.set_title(str(time).split('T')[0])

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()
            area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
            print(area_name)
            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates,"area_name":area_name,"type": analysis_type, "mean_res_rounded": mean_res_rounded, "labels": labels})
        except Exception as e:
            return jsonify({"error": e})
    return jsonify({"error": "Invalid method: "+request.method})
@app.route('/datasets', methods=['GET'])
def datasets():
    dc = datacube.Datacube(app='datacube-example')
    product_name = ['s2a_sen2cor_granule']

    p = []

    for j in product_name:
        datasets = dc.find_datasets(product=j)
        d = []
        if len(datasets) == 0:
            print('No datasets found for the specified query parameters.')
        else:
            for i in datasets:
                ds_loc = i.metadata_doc['geometry']['coordinates']
                d.append(ds_loc)
        unique_list = [x for i, x in enumerate(d) if x not in d[:i]]
        p+=unique_list
    unique_list = [x for i, x in enumerate(p) if x not in p[:i]]
    print(unique_list)
    return jsonify({'coordinates': unique_list})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')