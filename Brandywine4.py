import pandas as pd
import numpy as np
from statistics import median_grouped, mean
from collections import OrderedDict
from scipy import signal

PATH = "/home/markkhusidman/Desktop/Brandywine/"

def clean_data():

    df_usgs = pd.DataFrame()
    names = {1: "Downingtown", 3: "Wagontown"}
    for x in range(1, 4, 2):
        df = pd.read_csv(PATH + "USGS_Precip_raw" + str(x) + ".txt", skiprows=[28], usecols=[2, 4], index_col=0,
                         header=0, parse_dates=True, comment="#", names=["datetime", names[x]], sep="\t")
        df = df[~df.index.duplicated()]
        df = df.resample("D").sum()
        df_usgs = pd.concat([df_usgs, df], axis=1)

    df_usgs.to_csv(PATH + "Brandywine_USGS_test.txt")
    df_usgs.fillna(0, inplace=True)
    df_usgs = df_usgs.shift(-1)
    df_usgs.dropna(inplace=True)
    df_usgs.to_csv(PATH + "Brandywine_rain_test.txt")

    df = pd.read_csv(PATH + "Brandywine_USGS_EB.txt", usecols=[2, 4], index_col=0, parse_dates=True, delimiter="\t",
                     comment="#", skiprows=[28], header=0, names=["datetime", "Gage_Height"])

    df = df[~df.index.duplicated()]
    df = df.resample("15T").asfreq()
    df.interpolate(method="time", inplace=True)
    df.to_csv(PATH + "Brandywine_GHO.csv")

    df_ref = df.resample("D").asfreq()
    df_ref = df_ref.loc[pd.Timestamp("2014-03-14"): pd.Timestamp("2020-09-19"), :]
    df_ref.to_csv(PATH + "Brandywine_ref.csv")

    df["Gage_Height"] = signal.savgol_filter(df["Gage_Height"].values, 15, 2)
    df = df.resample("D").asfreq()
    df = df.loc[pd.Timestamp("2014-03-14"): pd.Timestamp("2020-09-20"), :]
    df.to_csv(PATH + "Brandywine_GHO_D_EB2.csv")

    df_final = pd.concat([df, df_usgs], axis=1)
    df_final.dropna(inplace=True)
    rain_temp = df_final.iloc[:, 1] + df_final.iloc[:, 2]

    intervals = [x for x in range(0, len(df), 45)]
    for item in df_final.columns:
        df_final[item] =  signal.detrend(df_final[item], bp=intervals)

    # df_temp = pd.DataFrame()
    # for i in range(len(df_final.columns)):
    #     item = df_final.iloc[:, i]
    #     new_item = []
    #     for x in range(len(item.values) - 45):
    #         interval = [0 + x, 45 + x]
    #         sl = df_final.iloc[interval[0]: interval[1], i]
    #         detrend = signal.detrend(sl)
    #         new_item.append(detrend[-1])
    #     df_temp[str(i)] = new_item
    #
    # df_final = df_final.iloc[45:, :]
    # rain_temp = rain_temp[45:]
    # for z in range(len(df_final.columns)):
    #     df_final.iloc[:, z] = df_temp.iloc[:, z].values

    df_final["All_Rain"] = np.gradient(rain_temp)
    df_final["Med_Diff"] = med_apply(df["Gage_Height"])
    df_final["Deriv"] = np.gradient(df_final["Gage_Height"])
    df_final["Gage_Diff"] = med_apply(df_final["Deriv"])
    df_final = df_final.applymap(lambda x: round(x, 3))
    df_final.to_csv(PATH + "Brandywine_test_EB2.csv")


def med_apply(data, simple=False):

    if simple:
        med = median_grouped(data)
        data = data - med
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

# def line_fit():
#     for m in range(Nreg):
#         Npts = bp[m + 1] - bp[m]
#         A = np.ones((Npts, 2), dtype)
#         A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
#         sl = slice(bp[m], bp[m + 1])
#         coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
#         newdata[sl] = newdata[sl] - np.dot(A, coef)

if __name__ == "__main__":

    clean_data()
