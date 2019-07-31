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
pip install Flask
conda install -c plotly plotly 

Then run the below command to download relevant nltk libraries:
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

After the installations, follow the below instruction using Python versions 3.*.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:8080/

Alternatively, you can use the trained model available in the 'models' folder and skip step 1.

## Project Motivation<a name="motivation"></a>

Typically, when there is a disaster, millions of messages seeking relief come via direct or social media. The primary challenge for disaster response organizations then becomes to filter out the relevant and most important messages from these. Usually, for disaster response management, different organizations handle different relief operations. While some of them might take care of water-related problems, some others may cater to shelter or medical supplies. So, it is imperative to categorize the relevant messages further into different categories to achieve better efficiency in relief operations. Figure-eight compiled a data set containing real messages sent during disaster events and categorized them under 36 different categories. Our aim for this project is to use natural language processing techniques to process this data and then train a supervised machine learning algorithm to classify disaster messages to categories so we can appropriately handle different needs of the affected people. This project also includes a web app where an emergency worker can input a new message and get classification results in several categories.

## File Descriptions <a name="files"></a>
There are three python scripts each deals with a specific part of the project.
Data:
* disaster_messages.csv: File containing 26248 messages from social media, news or direct sources.  
* disaster_categories.csv: File containing the category labels of the disaster messages. A message can belong to multiple categories.
* process_data.py: This is basically an ETL pipeline, where we cleaned data and saved in a database.
* DisasterResponse.db: Output of process_data.py script. It's a database file where the script stores data.
Model:
* train_classifier.py: This is where the text were processed using NLP techniques and then a ML pipeline was trained on the data.
* classifier.pkl: Output of train_classifier.py script. It contains the trained model that can be used for making predictions.
App:
* run.py: Launches the web app which takes a message as an input and outputs the categories it belongs to.
* templates: Contains the code for rendering the app in a web page.
Images: Contains images used in this description file.

## Results<a name="results"></a>
![Result 1](https://raw.githubusercontent.com/samardolui/DisasterResponse/master/images/dis_res1.jpg)
![Result 2](https://raw.githubusercontent.com/samardolui/DisasterResponse/master/images/dis_res2.jpg)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data was obtained from  [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). All credits to them for compiling the dataset and making it publicly available.
