import influxdb
import pandas as pd
import numpy as np
from scipy import signal
from ast import literal_eval
from datetime import timezone
from datetime import timedelta
from datetime import date
from datetime import datetime

PATH = "/home/markkhusidman/Desktop/Brandywine/"

# df_usgs = pd.DataFrame()
# names = {1: "Downingtown", 3: "Wagontown"}
# for x in range(1, 4, 2):
#     df = pd.read_csv(PATH + "USGS_Precip_raw" + str(x) + ".txt", skiprows=[28], usecols=[2, 4], index_col=0,
#                      header=0, parse_dates=True, comment="#", names=["datetime", names[x]], sep="\t")
#     df = df[~df.index.duplicated()]
#     df = df.resample("D").sum()
#     df_usgs = pd.concat([df_usgs, df], axis=1)
# df_usgs.to_csv(PATH + "influx_test1.txt")


# client = influxdb.DataFrameClient(host='localhost', port=8086)
# client.delete_series("training_raw", "Gage_Height")
# client.delete_series("training_raw", "Precip")
#
df_all = pd.read_csv(PATH + "Brandywine_test_EB2.csv", index_col=0, parse_dates=True)
#
# client.write_points(df_all.loc[:, ["Gage_Height"]], "Gage_Height", database="training")
# client.write_points(df_all.loc[:, ["Med_Diff"]], "Gage_Height", database="training")
# client.write_points(df_all.loc[:, ["Gage_Diff"]], "Gage_Height", database="training")
# client.write_points(df_all.loc[:, ["Deriv"]], "Gage_Height", database="training")
# client.write_points(df_all.loc[:, ["Downingtown"]], "Precip", database="training")
# client.write_points(df_all.loc[:, ["Wagontown"]], "Precip", database="training")
# client.write_points(df_all.loc[:, ["All_Rain"]], "Precip", database="training")

client = influxdb.InfluxDBClient(host='localhost', port=8086)
result1 = client.query("select * from Gage_Height", database="training_raw")
print(result1)

# result2 = client.query("select * from Precip", database="training")
# idx = pd.Timestamp(date.strftime(datetime.now(timezone(-timedelta(hours=5))) + timedelta(days=1), "%Y-%m-%d"))
# client = influxdb.InfluxDBClient(host='34.231.230.82', port=8086, username="datasci",
#                                  password="pekosoftdatascience2020", database="peko_prediction_results")
# client.delete_series(measurement="prediction")
# idx = [pd.Timestamp("2020-10-09"), pd.Timestamp("2020-10-10"), pd.Timestamp("2020-10-11")]
# df = pd.DataFrame([2.307810945488868, 2.282817544203472, 2.2523350038639465], index=idx, columns=["Gage_Height"])
# client = influxdb.DataFrameClient(host='34.231.230.82', port=8086, username="datasci",
#                                   password="pekosoftdatascience2020", database="peko_prediction_results")
# #
# client.write_points(df, "ppred", database="peko_prediction_results")
client = influxdb.InfluxDBClient(host='34.231.230.82', port=8086, username="datasci",
                                 password="pekosoftdatascience2020", database="peko_prediction_results")
# client = influxdb.InfluxDBClient(host='34.231.230.82', port=8086, username="datasci", password="pekosoftdatascience2020",
#                                  database="pekoiotlevelmaxdb")
# print(client.get_list_measurements())
result = client.query("select * from ppred where time < now() + 1d")
print(result)
# points = result.get_points(tags={"filenameref": "2020-10-04_00_00"})
# for point in points:
#     print(point["timestamp"])
#     print(point["maxy"])


        # points = list(result1.get_points(measurement="Gage_Height"))
# points = points[-4:]
# df = pd.DataFrame()
# for item in result1["Gage_Height"]:
#     key_list = []
#     val_list = []
#     raw_vals = item.items()
#     for key, val in raw_vals:
#         key_list.append(key)
#         val_list.append(val)
#     idx_val = pd.Timestamp(val_list[0])
#     for x in range(1, len(key_list)):
#         df.loc[idx_val, key_list[x]] = val_list[x]
#
# for item in result2["Precip"]:
#     key_list = []
#     val_list = []
#     raw_vals = item.items()
#     for key, val in raw_vals:
#         key_list.append(key)
#         val_list.append(val)
#     idx_val = pd.Timestamp(val_list[0])
#     for x in range(1, len(key_list)):
#         df.loc[idx_val, key_list[x]] = val_list[x]
#
# df = df.applymap(lambda x: round(x, 3))
# col_order = ["Gage_Height", "Downingtown", "Wagontown", "All_Rain", "Med_Diff", "Deriv", "Gage_Diff"]
# df = df[col_order]
# df.to_csv(PATH + "influx_test6.txt")