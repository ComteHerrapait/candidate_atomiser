# candidate_atomiser
Name chosen for our Big Data project, project for Télécom Saint-Etienne.  
The members of our team are :
* Victor Archambault
* Léon Delcroix
* Emilion Goddet

# Project presentation
The goal of the project is to create a complete environment to collect data, upload it to the cloud, process it thanks to a trained model and gather the results in order to put them in a mongoDB database. The project contains 6 steps in total.  
The project contain three files :
* data.json : json file containing informations about 217,197 individuals (Id, description, gender)
* label.csv : contains the job associated to every individual
* categories_string.csv : contains the link between a job's id and its description

## Step 0
Upload all three files to a Hadopp VM, which is installed on Virtual Box.

## Step 1
Download the files on your computer thanks to a dedicated script.

## Step 2
Upload the data on AWS. The transfer needs to be secured, and the files encrypted.

## Step 3
Create a training model on the AWS VM thanks to machin learning and some of the data uploaded.

## Step 4
Create a file called predict.csv containing some data (not the ones used to train the model). Then, a script will be executed to predict the job of the individuals in predict.csv using the model. The result then needs to be saved in a csv file, result.csv in our case.

## Step 5
Download on your computer result.csv, then upload its content on a mongoDB database thanks to a dedicated script. 

# Chosen technologies

For this project, our main tool was Python. Every script was written in Python, it is the best language for machine learning, and it is very easy to upload or download files using scp.  
MongoDB was chosen for the final step because it is constructed like a json file, and so is perfect to store the results.  
For the training model, we chose to use SVC because it combined speed and good precision.  
We used many libraries to complete the project, which can all be installed using pip :
* pandas : Main library to read and process data
* sklearn : Used to create the model
* nltk : Used to process text
* cryptography : Used to encrypt and decrypt files
* pymongo : Used to send the results to the mongoDB database
* configparser : Used to read easily a config file
* paramiko : Used for ssh and scp connections

# Setup
run `pip install -r requirements.txt` to install dependencies  
connect.conf needs to be changed to adapt to a specific environment. Currently it contains informations to work in our specific case.

# Warning
Please do not push code containing authentification info, use the config files and keep them local.

# ERRROR : resource stopwords not found.
if you get this error, please run this : `nltk.download('stopwords')`