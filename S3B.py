import boto3
import usgsAPI2 as uapi
import wuAPI as wapi
from datetime import timezone
from datetime import timedelta
from datetime import date
from datetime import datetime

PATH = "/home/markkhusidman/Desktop/Brandywine/"

s3 = boto3.resource("s3")
for bucket in s3.buckets.all():
    print(bucket.name)

site_list = ["Downingtown", "Wagontown"]

def collect_wu_data(site_list):

    tod = date.strftime(datetime.now(timezone(-timedelta(hours=5))), "%Y-%m-%d")
    for site in site_list:
        wapi.collect_data(PATH, site, tod)
        df = wapi.wu_parse(PATH + "forecast_%s_%s.txt" % (site, tod), tod, False, "both")
        df = df.iloc[0, :]
        hour = df.name.hour
        minute = df.name.minute
        o = open(PATH + "wu_precip_%s_%s-%s-%s" % (site, tod, hour, minute), "w+")
        o.write("%d\t%d" % (df.values[0], df.values[1]))
        o.close()


collect_wu_data(sites)