# Disaster Response Pipeline Project
## UDACITY - Data Science Nanodegree

Table of Content
1. [Description](#description)
2. [Project Motivation](#Motivation)
2. [Components](#project_components)
    - [ETL Pipeline](#etl)
    - [ML Pipeline](#ml_pipeline)
    - [Flask Web App](#flask)
3. [Getting Started](#getting_started)
    - [Dependencies](#dependencies)
    - [Installing](#installing)
    - [Instructions](#instructions)
4. [File Description](#file)
5. [Licensing, Authors, and Acknowledgements](#Licensing)

<a name="description"></a>
## Description
This is my second repo for the "Disaster Response Pipeline" project in Udacity Data Scientist Nano Degree Program. In collaboration with Figure Eight, this project used datasets containing pre-labeled tweets and messages from real-life disaster events.

<a name="Motivation"></a>
## Project Motivation
The motivation behind this project is to potentially build an app to classify real disaster messages. 
This project is divided in the following key sections:
- Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
- Build a machine learning pipeline to train the which can classify text message in various categories
- Run a web app which can show model results in real time

<a name="Motivation"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# Disaster-Response-Pipeline
