import requests
import pandas as pd

PATH = "/home/markkhusidman/Desktop/Brandywine/"

def collect_data(PATH, typ, site_no, start, end, p=False):

    if typ == "rain":
        param = "00045"
    elif typ == "gage":
        param = "00065"
    else:
        raise ValueError("Invalid Data Type")

    url = "https://waterdata.usgs.gov/nwis/uv?search_site_no=" + site_no + "&search_site_no_match_type=exact&" \
          "index_pmcode_" + param + "=1&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=" \
          "site_no&column_name=station_nm&range_selection=date_range&begin_date=" + start + "&end_date=" + end + \
          "&format=rdb&date_format=YYYY-MM-DD&rdb_compression=value&list_of_search_criteria=search_site_no%2Crealtime" \
          "_parameter_selection"

    response = requests.get(url)
    data = response.text

    if p:
        print(data)
    o = open(PATH + "%s_%s_%s.txt" % (typ, site_no, end), "w+")
    o.write(data)
    o.close()

def parse_data(file):
    df = pd.read_csv(file, skiprows=[27], usecols=[2, 4], index_col=0,
                     header=0, parse_dates=True, comment="#", names=["datetime", "data"], sep="\t")
    df = df[~df.index.duplicated()]
    if "rain" in file:
        df = df.resample("D").sum()
    elif "gage" in file:
        df = df.resample("D").asfreq()
    else:
        raise ValueError("Invalid File Name")
    return df


if __name__ == "__main__":

    collect_data(PATH, "gage", "01480870", "2020-10-10", "2020-10-10")
    collect_data(PATH, "rain", "01480870", "2020-10-10", "2020-10-10")
    collect_data(PATH, "rain", "01480399", "2020-10-10", "2020-10-10")
    df1 = parse_data(PATH + "gage_01480870_2020-10-10.txt")
    df2 = parse_data(PATH + "rain_01480870_2020-10-10.txt")
    df3 = parse_data(PATH + "rain_01480399_2020-10-10.txt")
    df_final = pd.concat([df1, df2], axis=1)
    df_final = pd.concat([df_final, df3], axis=1)
    # df_final.iloc[:, 1:] = df_final.iloc[:, 1:].shift(1)
    print(df_final)
