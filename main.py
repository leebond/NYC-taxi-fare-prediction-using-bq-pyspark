#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:11:53 2020

@author: leebond
"""
from __future__ import print_function

import sys
from pyspark.sql.functions import abs, to_timestamp
from pyspark import SparkContext, SparkConf, SQLContext
from datetime import datetime
from pyspark.sql.session import SparkSession

def getDataForLocal(limit):
    from google.cloud import bigquery

    client = bigquery.Client()
    
    query = """
    SELECT fare_amount , trip_distance  FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016`
    WHERE
    fare_amount >= 2.5 AND
    trip_distance > 1 AND trip_distance < 1000 AND
    TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) > 0 AND
    TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND)/3600 <= 24 AND
    dropoff_latitude != 0 AND dropoff_longitude != 0 AND   
    pickup_latitude != 0 AND pickup_longitude != 0
    
    """

    query = query + "limit " + str(limit)
    
    return client.query(query)

def getDataForDataproc():
    '''
    A different method is required to read data from BigQuery from Dataproc as Dataproc does not have
    google-cloud python package preinstalled. Even after attempting to install the package via the
    via the SSH console, it still doesn't work
    '''
    # Read the data from BigQuery as a Spark Dataframe.
    df = spark.read.format("bigquery").option("table", "bigquery-public-data.new_york.tlc_yellow_trips_2016").load()
    df = df.filter(df.fare_amount > 2.5)
    df = df.filter(df.trip_distance > 0)
    df = df.filter(df.trip_distance < 1000)
    df = df.filter(df.dropoff_latitude != 0)
    df = df.filter(df.dropoff_longitude != 0)
    df = df.filter(df.pickup_latitude != 0)
    df = df.filter(df.pickup_longitude != 0)
#    to_timestamp(df.t, 'yyyy-MM-dd HH:mm:ss')
#    df = df.filter(to_timestamp(df.dropoff_datetime, ) > to_timestamp(df.pickup_datetime)
#    df = df.filter((to_timestamp(df.dropoff_datetime, ) - to_timestamp(df.pickup_datetime, )/3600 <=24 )

    return df

def run(n, mode):
    if mode == '--small':
        data = getDataForLocal(n).to_dataframe() # to_dataframe() converts a QueryJob object to a Pandas dataframe
#        print(data.shape)
        spark_df = sqlContext.createDataFrame(data).cache() # createDataFrame converts a Pandas dataframe to Spark dataframe()
    elif mode == '--large':
        spark_df = getDataForDataproc()
        n = spark_df.count()
    
    
    ## create sum_xy and sum_xx terms -> these are terms needed in OLS calculation
    spark_df = spark_df.select("fare_amount", "trip_distance", (spark_df.trip_distance*spark_df.fare_amount).alias("distxfare"),\
                               (spark_df.trip_distance*spark_df.trip_distance).alias("distxdist"))
    
    sum_all = spark_df.groupBy().sum().collect()
    
    sum_y, sum_x, sum_xy, sum_xx = sum_all[0]
    
    beta_1 = (sum_xy - (sum_x*sum_y)/n)/(sum_xx - (sum_x**2)/n)
    beta_0 = sum_y/n - beta_1*sum_x/n
    
    y_mean = sum_y/n

    ## save the prediction and check prediction performance metrics
    spark_df = spark_df.withColumn("OLSPred", spark_df.trip_distance*beta_1+beta_0 )
    spark_df = spark_df.withColumn("r_abs", abs(spark_df.OLSPred-spark_df.fare_amount) )
    spark_df = spark_df.withColumn("r_sq", (spark_df.OLSPred-spark_df.fare_amount)**2 )
    spark_df = spark_df.withColumn("y_var", (spark_df.fare_amount - y_mean)**2)
    sum_all = spark_df.groupBy().sum().collect()
    sum_y, sum_x, sum_xy, sum_xx, OLSPred_sum, r_abs_sum, r_sq_sum, y_var_sum = sum_all[0]
        
#    print(spark_df.show(2))
    print("With a dataset of training size %s," %n)
    print("beta_1: %s, beta_0: %s " %(beta_1, beta_0))
    print("MSE: %s, MAE: %s, R2: %s" %(r_sq_sum/n, r_abs_sum/n, 1-r_sq_sum/y_var_sum))
    

if __name__=='__main__':
    if len(sys.argv) != 2:
        print("Usage: python <python-file.py> <mode>", file=sys.stderr)
        exit(-1)

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    spark = SparkSession(sc)
    
    mode = str(sys.argv[1])

    if mode == '--small':
        n = 100000
        st = datetime.now()
        run(n, mode)
        elapse = datetime.now() - st
        print("Time taken to run small dataset: %s mins %s secs" %(elapse.seconds//60, elapse.seconds))

    
    elif mode == '--large':
        n = ''
        st = datetime.now()
        run(n, mode)
        elapse = datetime.now() - st
        print("Time taken to run small dataset: %s mins %s secs" %(elapse.seconds//60, elapse.seconds))
        
    sc.stop()    