#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:11:53 2020

@author: leebond
"""
import sys
from pyspark.sql.functions import abs
from pyspark import SparkContext, SparkConf, SQLContext

def getData(limit):
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


    

if __name__=='__main__':
#    if len(sys.argv) != 2:
#        print("Usage: python <python-file.py> <mode>", file=sys.stderr)
#        exit(-1)
#    
#    if str(sys.argv[1]) == '--small':
#        
#        print('running small dataset')
#        ##todo read small dataset, then run linear model
#    elif str(sys.argv[1]) == '--big':

    n = 1000
    data = getData(n).to_dataframe() # converts a QueryJob object to a Pandas dataframe
    print(data.shape)
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc) 
    
    spark_df = sqlContext.createDataFrame(data).cache()
    
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
    
    
    sc.stop()    