# Disaster Response Pipeline Project
## UDACITY - Data Science Nanodegree

### Table of Content
1. [Description](#description)
2. [Project Motivation](#Motivation)
3. [Folder Discription](#Folder)
4. [Getting Started](#getting)
    - [Libraries](#libraries)
    - [Instructions](#instructions)
4. [Results](#result)
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

<a name="Folder"></a>
## Folder Discription

         disaster-response-pipeline
          |-- app
                |-- templates
                        |-- go.html # main page of web app
                        |-- master.html # classification result page of web app
                |-- run.py # Flask file that runs app
          |-- data                
                |-- DisasterResponse.db # clean database 
                |-- categories.csv # data to process 
                |-- message.csv # data to process
                |-- process_data.py
          |-- models
                |-- classifier.rar (classifier.pkl) # saved model in pickle
                |-- train_classifier.py
          |-- image     
          |-- README

<a name="getting"></a>
## Getting Started

<a name="libraries"></a>
### Libraries
The libraries I used for this project were:
- Python 3.
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly
- etc

You will also need to have software installed to run and execute an iPython Notebook

<a name="instructions"></a>
### Instructions:
    1. Run the following commands in the project's root directory to set up your database and model.

          - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
          - To run ML pipeline that trains classifier and saves
             `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    2. Run the following command in the app's directory to run your web app.
         `python run.py`

    3. Go to http://0.0.0.0:3001/

## Results<a name="result"></a>
Below are a few screenshots of the web app.
![web app 1](https://github.com/Sesil01/Udacity---Disaster-Pipeline-Project/blob/main/Image/web%20app%201.png?raw=true)
![web app 2](https://github.com/Sesil01/Udacity---Disaster-Pipeline-Project/blob/main/Image/web%20app%202.png?raw=true)
![web app 3](https://github.com/Sesil01/Udacity---Disaster-Pipeline-Project/blob/main/Image/web%20app%203.png?raw=true)

## Licensing, Authors, and Acknowledgements<a name="Licensing"></a>
-

## Thanks to 
![logo.png](http://pra-dan.github.io/img/udacimak/logo.png)
![Attached_to_figure-eight-dot-com](https://upload.wikimedia.org/wikipedia/en/a/a6/Attached_to_figure-eight-dot-com.png)
