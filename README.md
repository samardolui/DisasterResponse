# Disaster Response
![alt text](https://raw.githubusercontent.com/samardolui/DisasterResponse/master/images/cover_image.jpg)
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Besides the Anaconda distribution of Python, NLTK and sqlalchemy libraries needs to be installed.
Use below commands to install these two libraries.

pip install nltk
pip install SQLAlchemy

After the installations, follow the below instruction using Python versions 3.*.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pk`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:8080/

If you want to use the existing trained model instead, then you may want to skip step 1 and step 2.

## Project Motivation<a name="motivation"></a>

Typically, when there is a disaster, millions of messages seeking relief come via direct or social media. The primary challenge for disaster response organizations then becomes to filter out the relevant and most important messages from these. Usually, for disaster response management, different organizations handle different relief operations. While some of them might take care of water-related problems, some others may cater to shelter or medical supplies. So, it is imperative to categorize the relevant messages further into different categories to achieve better efficiency in relief operations. Figure-eight compiled a data set containing real messages sent during disaster events and categorized them under 36 different categories. Our aim for this project is to use natural language processing techniques to process this data and then train a supervised machine learning algorithm to classify disaster messages to categories so we can appropriately handle different needs of the affected people. This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. 

## File Descriptions <a name="files"></a>
There are three python scripts each deals with a specific part of the project.
1. process_data.py:  This is basically an ETL pipeline, where we clean data and load into database.
2. train_classifier.py:  ML pipeline for training a classifier and saving the model into a file.
3. run.py: Launches the web app where we can input a message and get the categories it belongs to.

## Results<a name="results"></a>
![alt text](https://raw.githubusercontent.com/samardolui/DisasterResponse/master/images/dis_res1.PNG)
![alt text](https://raw.githubusercontent.com/samardolui/DisasterResponse/master/images/dis_res2.PNG)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data was obtained from  [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). All credits to them for compiling the dataset and making it publicly available.
