import influxdb
import pandas as pd
import numpy as np
from scipy import signal
from Classy2 import Model
import usgsAPI2 as uapi
import wuAPI as wapi
from datetime import timezone
from datetime import timedelta
from datetime import date
from datetime import datetime
import time
from statistics import median_grouped, mean
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
from tensorflow import keras


PATH = "/home/markkhusidman/Desktop/Brandywine/"

def update_table(start, end):

    g_start = start
    g_end = end
    r_start = date.strftime(pd.Timestamp(start) - pd.Timedelta(days=1), "%Y-%m-%d")
    r_end = date.strftime(pd.Timestamp(end) - pd.Timedelta(days=1), "%Y-%m-%d")
    uapi.collect_data(PATH, "gage", "01480870", g_start, g_end)
    uapi.collect_data(PATH, "rain", "01480870", r_start, r_end)
    uapi.collect_data(PATH, "rain", "01480399", r_start, r_end)

    df1 = uapi.parse_data(PATH + "gage_01480870_%s.txt" % g_end)
    df1.rename(columns={"data": "Gage_Height"}, inplace=True)
    df2 = uapi.parse_data(PATH + "rain_01480870_%s.txt" % r_end)
    df2.rename(columns={"data": "Downingtown"}, inplace=True)
    df3 = uapi.parse_data(PATH + "rain_01480399_%s.txt" % r_end)
    df3.rename(columns={"data": "Wagontown"}, inplace=True)

    client = influxdb.DataFrameClient(host='localhost', port=8086)
    client.write_points(df1, "Gage_Height", database="training_raw")
    client.write_points(df2, "Precip", database="training_raw")
    client.write_points(df3, "Precip", database="training_raw")


def collect_wu_data():

    w_start = date.strftime(datetime.now(timezone(-timedelta(hours=5))) + timedelta(days=1), "%Y-%m-%d")
    w_end = date.strftime(datetime.now(timezone(-timedelta(hours=5))) + timedelta(days=2), "%Y-%m-%d")
    wapi.collect_data(PATH, "Downingtown", w_start)
    wapi.collect_data(PATH, "Downingtown", w_end)
    wapi.collect_data(PATH, "Wagontown", w_start)
    wapi.collect_data(PATH, "Wagontown", w_end)

    df1 = wapi.wu_parse(PATH + "forecast_Downingtown_%s.txt" % w_start, w_start)
    df2 = wapi.wu_parse(PATH + "forecast_Downingtown_%s.txt" % w_end, w_end)
    df3 = wapi.wu_parse(PATH + "forecast_Wagontown_%s.txt" % w_start, w_start)
    df4 = wapi.wu_parse(PATH + "forecast_Wagontown_%s.txt" % w_end, w_end)

    return df1, df2, df3, df4

def compile_raw_input(df1, df2, df3, df4):

    client = influxdb.InfluxDBClient(host='localhost', port=8086)
    result1 = client.query("select * from Gage_Height", database="training_raw")
    result2 = client.query("select * from Precip", database="training_raw")
    df_final = df_from_query(result1, result2)

    w_start = date.strftime(datetime.now(timezone(-timedelta(hours=5))), "%Y-%m-%d")
    w_end = date.strftime(datetime.now(timezone(-timedelta(hours=5))) + timedelta(days=1), "%Y-%m-%d")
    df_final.loc[pd.Timestamp(w_start, tz="UTC"), "Downingtown"] = df1.values[0]
    df_final.loc[pd.Timestamp(w_end, tz="UTC"), "Downingtown"] = df2.values[0]
    df_final.loc[pd.Timestamp(w_start, tz="UTC"), "Wagontown"] = df3.values[0]
    df_final.loc[pd.Timestamp(w_end, tz="UTC"), "Wagontown"] = df4.values[0]

    df_final.iloc[:, 1:] = df_final.iloc[:, 1:].shift(-1)
    df_final.dropna(inplace=True)
    df_final.to_csv(PATH + "fc_test1.txt")

    return df_final


def modify_input(raw_input):

    df = raw_input
    gage_ref = df.iloc[:, 0].copy()
    rain_temp = df.iloc[:, 1] + df.iloc[:, 2]
    rt_val = None

    intervals = [x for x in range(0, len(df), 45)]
    for item in df.columns:
        df[item] = signal.detrend(df[item], bp=intervals)
        if item == "Gage_Height":
            trend = gage_ref - df[item]
            slope = trend[-1] - trend[-2]
            rt_val = trend[-1] + slope

    df["All_Rain"] = np.gradient(rain_temp)
    df["Med_Diff"] = med_apply(df["Gage_Height"])
    df["Deriv"] = np.gradient(df["Gage_Height"])
    df["Gage_Diff"] = med_apply(df["Deriv"])
    df = df.applymap(lambda x: round(x, 3))
    df.to_csv(PATH + "fc_test2.txt")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)
    scaled = scaled[-5:, :]

    df2 = Model.series_to_supervised(scaled, 4, 1)
    to_drop = [x for x in range((df.shape[1] * 4) + 1,
                                 (df.shape[1]) * (4 + 1))]
    df2.drop(df2.columns[to_drop], axis=1, inplace=True)
    df2 = df2.iloc[:, :-1]
    df2 = df2.applymap(lambda x: round(x, 3))
    vals = df2.values
    vals = vals.reshape((vals.shape[0], 4, int(vals.shape[1] / 4)))


    return vals, scaler, rt_val


def predict(input, scaler, rt_val, test=False):

    model = keras.models.load_model(PATH + "Model_EB3_2434.h5")
    raw_forecast = model.predict(input)
    to_trans = np.zeros(7)
    to_trans[0] = raw_forecast
    to_trans = to_trans.reshape(1, len(to_trans))
    transform = scaler.inverse_transform(to_trans)
    transform = transform[0][0]
    forecast = transform + rt_val

    if not test:
        idx = pd.Timestamp(date.strftime(datetime.now(timezone(-timedelta(hours=5))) + timedelta(days=1), "%Y-%m-%d"))
        df = pd.DataFrame(forecast, index=[idx], columns=["Gage_Height"])
        client = influxdb.DataFrameClient(host='34.231.230.82', port=8086, username="datasci",
                                          password="pekosoftdatascience2020", database="peko_prediction_results")
        client.write_points(df, "ppred", database="peko_prediction_results")

    else:
        return forecast


def df_from_query(q1, q2):

    df = pd.DataFrame()
    for item in q1["Gage_Height"]:
        key_list = []
        val_list = []
        raw_vals = item.items()
        for key, val in raw_vals:
            key_list.append(key)
            val_list.append(val)
        idx_val = pd.Timestamp(val_list[0])
        for x in range(1, len(key_list)):
            df.loc[idx_val, key_list[x]] = val_list[x]

    for item in q2["Precip"]:
        key_list = []
        val_list = []
        raw_vals = item.items()
        for key, val in raw_vals:
            key_list.append(key)
            val_list.append(val)
        idx_val = pd.Timestamp(val_list[0])
        for x in range(1, len(key_list)):
            df.loc[idx_val, key_list[x]] = val_list[x]

    col_order = ["Gage_Height", "Downingtown", "Wagontown"]
    df = df[col_order]

    return df


def med_apply(data, simple=False):

    if simple:
        med = median_grouped(data)
        data -= med
        return data

    d = historic_med()
    ns = pd.Series(index=data.index, dtype='float64')
    for item in data.items():
        date = item[0].month
        ns.loc[item[0]] = item[1] - d[date]
    return ns


def historic_med():

    d = dict()
    df = pd.read_csv(PATH + "Brandywine_GHO.csv", index_col=0, parse_dates=True)
    rs = df["Gage_Height"].resample("M", label="left", closed="left")
    df = rs.agg(median_grouped)
    for item in df.items():
        date = item[0].month
        if date in d.keys():
            d[date].append(item[1])
        else:
            d[date] = [item[1]]
    for key, vals in d.items():
        d[key] = round(mean(vals), 3)
    return d



df1, df2, df3, df4 = collect_wu_data()
time.sleep(5700)
cd = date.strftime(datetime.now(timezone(-timedelta(hours=5))), "%Y-%m-%d")
update_table(cd, cd)
input = compile_raw_input(df1, df2, df3, df4)
# df = pd.read_csv(PATH + "fc_test1.txt", index_col=0, parse_dates=True)
m, s, r = modify_input(input)
print(predict(m, s, r))

