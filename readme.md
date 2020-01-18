# NYC Taxi Fare Prediction using Simple Linear Regression with BigQuery and PySpark

### Description
Running a Simple Linear Regression on a regression problem might not be the state-of-the-art way to learn parameters. However, when the volumn of data increases drastically, state-of-the-art algorithms will not be able to learn parameters quick enough even with the best computing resources.

As such, in this repo, I attempt the use of a simple linear regression to predict NYC taxi fares which has over 100 million records per year in a single SQL table.

The data is stored in BigQuery and PySpark would be the tool to go for such big data tasks.

### How to use
run main.py with python and give a mode parameter with tells the code to run on a limited sized dataset or the full SQL table <br/>
`> python main.py <mode>` <br/>
[mode]: 
--small runs a dataset of size 10,000 nyc taxi records<br/>
(DO NOT RUN ON LOCAL) --large runs the full SQL table `bigquery-public-data.new_york.tlc_yellow_trips_2016` of 131,165,043 nyc taxi records<br/>
*Setup a GCP Dataproc Cluster to run on the full dataset.*

### Setup Requirements
1. Get a Google Cloud Platform account

2. Create a GCP project or use an existing GCP project of yours<br/>
You will need a GCP project in order to run BigQuery api calls. Create a GCP project through your GCP console from https://console.cloud.google.com/project

3. Enable GCP BigQuery API on your aforementioned project
You will need to run BigQuery api calls through javascript using BigQuery's javascript package. Make sure you have enabled it to the project at https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com

4. Setup Authentication and a private key
GCP's Authentication allows your local machine to make authenticated api calls to GCP's services. You will need to create a private Service account key and set up your local machine to use the specified private key by adding it to your PATH by running the following command in your terminal
`export GOOGLE_APPLICATION_CREDENTIALS="[PATH to your .json private key]"` or add to your `./bash_profile`.
