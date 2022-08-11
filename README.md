# 

# 📓📓 Book-Recommendation-System 📓📓


## 📝 Overview 📝

A simple Book Recommendation System, built using Flask and deployed on Heroku.

## 💡 Motivation 💡

As being a Data and ML enthusiast I have tried many different projects related to the subject but what I have realised
is that Deploying your machine learning model is a key aspect of every ML and Data science project. Everything thing I 
had studied or been taught so far in my Data science and ML journey had mostly focused on defining problem statement followed by Data collection
and preparation, model building and evaluation process which is of course important for every ML/DS project but what if I want different people to
interact with my models, how can I make my model available for end-users? I can't send them jupyter notebooks right!. That's why I wanted to try my hands
on complete end-to-end machine learning project.

## 🎬 Deployed Application 🎬

To View the Deployed Application, click on the link given below : Books recommendation system Web App - https://bookrecommendation2022.herokuapp.com/




https://user-images.githubusercontent.com/63314917/184183089-64e5f93d-c109-4ac2-8d0e-42b965823f37.mp4






# Technical Aspect
This Project is mainly divided into two parts:

- Exploring the dataset and traning the model using Sklearn.
- Building and hosting a flask web app on Heroku.


## About the repository Structure :

- Project consist **app.py** script which is used to run the application and is engine of this app. contians API that gets input from the user and computes a predicted value based on the model.
- **app.py** contains code to build and train a Machine learning model.
- Templates folder contains two files **main.html** and **result.html** which describe the structure of the app and the way this web application behaves. These files are connected with Python via Flask framework.
- static folder contains file **style.css** which adds some styling and enhance the look of the application.





## Setup the environment
1. Create a new virtual environment 
2. Activate the new environment
3. Donwlnload the file requirement.txt  
4. Install the requirement 

```bash
$ pip install -r requirements.txt 

```
## To Run the Application

```bash
python app.py
```

## Deployement on Heroku
Install Heroku CLI as this makes it easy to create and manage your Heroku apps directly from the terminal. You can download it from [here](https://devcenter.heroku.com/articles/heroku-cli)

next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.
