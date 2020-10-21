from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip
import pandas as pd

PATH = "/home/markkhusidman/Desktop/Brandywine/"

def collect_data(PATH, site, date):

    o = Options()
    o.binary_location = "/usr/bin/google-chrome"
    driver = webdriver.Chrome('/home/markkhusidman/Desktop/chromedriver', chrome_options=o)
    url = "https://www.wunderground.com/hourly/us/pa/%s/date/%s" % (site, date)
    driver.get(url)

    html1 = driver.page_source
    html2 = driver.execute_script("return document.documentElement.innerHTML;")
    ActionChains(driver).pause(7).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).pause(1).\
        key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL).perform()
    data = pyperclip.paste()

    o = open(PATH + "forecast_%s_%s.txt" % (site, date), "w+")
    o.write(data)
    o.close()
    driver.quit()


def wu_parse(file, date, resamp=True, d_type="precip"):
    flag = False
    percent = ""
    precip = ""
    df = pd.DataFrame()
    o = open(file, "r")
    for line in o:
        if "Time	Conditions" in line:
            flag = True
            continue
        elif "NEXT DAY HOURLY FORECAST" in line:
            break
        if flag and line != "\n":
            splt = line.split()
            hour = splt[0].split(":")[0]
            if splt[1] == "pm" and hour != "12":
                hour = str(int(hour) + 12)
            elif splt[1] == "am" and hour == "12":
                hour = "0"
            time = hour + ":00"
            datetime = pd.to_datetime(date + " " + time)
            for x in range(len(splt)):
                if splt[x] == "%":
                    percent = splt[x - 1]
                    precip = splt[x + 1]
                    break
            df.loc[datetime, "Percent"] = percent
            df.loc[datetime, "Precip"] = precip

    df["Percent"] = df["Percent"].astype(float)
    df["Precip"] = df["Precip"].astype(float)
    if resamp:
        funcs = {"Percent": "mean", "Precip": "sum"}
        r = df.resample("D")
        df = r.agg(funcs)

    if d_type == "precip":
        return df.loc[:, ["Precip"]]
    elif d_type == "percent":
        return df.loc[:, ["Percent"]]
    elif d_type == "both":
        return df
    else:
        raise ValueError("Invalid d_type: choose 'precip', 'percent', or 'both'")


if __name__ == "__main__":
    collect_data(PATH, "wagontown", "2020-10-15")
    # collect_data(PATH, "downingtown", "2020-10-15")
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15"))
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15", d_type="both"))
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15", d_type="percent"))
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15", resamp=False))
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15", resamp=False, d_type="both"))
    print(wu_parse(PATH + "forecast_wagontown_2020-10-15.txt", "2020-10-15", resamp=False, d_type="percent"))
