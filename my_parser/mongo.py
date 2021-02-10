import csv
import json
import pandas as pd
import sys, getopt, pprint
import configparser
from pymongo import MongoClient

def import_data_mongo(conf_path="../config/connect.conf"):
    """
    Import result.csv from local to the mongoDB database.

    Args:
        conf_path (string): Path to the conf file, by default ../config/connect.conf
    """

    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    local_path = config['DEFAULT']['PATH_LOCAL_PROCESSED']

    #CSV to JSON Conversion
    csvfile = open('local_path'+'result.csv', 'r')
    reader = csv.DictReader( csvfile )
    mongo_client=MongoClient() 
    db=mongo_client.results
    db.segment.drop()
    header= [ "ID", "Description", "Gender", "Predicted Job"]

    for each in reader:
        row={}
        for field in header:
            row[field]=each[field]

        db.segment.insert(row)