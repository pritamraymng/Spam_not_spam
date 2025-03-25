#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os                                             
import numpy as np                                    
from sklearn.svm import SVC                           
from sklearn.model_selection import train_test_split  
import csv

#predefined set of keywords likely to indicate SPAM
spam_keywords= ['lottery', 'win', 'money', 'cash', 'prize', 'urgent', 'free', 'job', 'offer', 'buy', 'click', 'guarantee',
    'investment', 'urgent reply', 'unlimited', 'subscribe', 'hello', 'hiring', 'credit', 'claim', 'now', 'access', 'bonus']


#cleaning of data
def clean_text(text):                                 #converts text into lowercase,keeps letter, space, numbers
    return ''.join(char.lower() if char.isalnum() or char.isspace() else '' for char in text)  #replace others by a space


#loading and cleaning data
def load_data(data_file):
    data= []                                           #empty list to store text of each email
    labels= []                                         #empty list to store labels of each email
    with open(data_file, 'r', encoding= 'utf-8') as file:
        for line in file:                              #loop through each line
            try:
                text, label= line.strip().split(',')    #split each line in (text, label)
                label= int(label)                       #convert labels to integer from string 
                if label in [0,1]:
                    data.append(clean_text(text))      #if data is 0(non-spam), 1(spam) add text in the list 'data'
                    labels.append(label)               #add label in list 'labels' created earlier
            except ValueError:
                                                       #skip rows with missing values
                    continue
    return data, np.array(labels)               

#feature vector using frequency of each keyword
def extract_features(data):                               #function to construct feature matrix
    features = np.zeros((len(data), len(spam_keywords)))  #initialize a matrix of zeros
    for i, email in enumerate(data):                      #iterate over each email in data
        words= email.split()                              #split email into individual words
        for j, keyword in enumerate(spam_keywords):       #for each keyword in spam list, count occurance
            features[i, j]= words.count(keyword)
    return features                                       # matrix, each row is a feature vector


#Naive Bayes classifier

def calculate_prior(labels):                              #calculate prior of spam
    return np.sum(labels)/len(labels)                     #proportion of emails labeled as 1

def calculate_word_probs(features, labels):               #laplace smoothing function
    pseudo_email= np.ones((1, features.shape[1]))          #add email containing all words
                                                          #concatenate 2 pseudo emails in the feature set
    features= np.vstack([features, pseudo_email, pseudo_email])
    labels= np.append(labels, [1, 0])                     #add lebel 1 for spam pseudo email, label 0 for non-spam pseudo email
    
    spam_emails= features[labels==1]                      #seperate spam emails
    non_spam_emails= features[labels==0]                  # non-spam emails
    
    spam_word_counts= spam_emails.sum(axis=0)             #spam keyword count in spam email(summation along column, axix=0)
    non_spam_word_counts = non_spam_emails.sum(axis=0)    #non-spam keyword count in non-spam email
                                                          #calculate conditional probabilities
        
    spam_word_probs= spam_word_counts/spam_word_counts.sum()                 #probability of each word given spam
    non_spam_word_probs = non_spam_word_counts / non_spam_word_counts.sum()  # non-spam
    
    return spam_word_probs, non_spam_word_probs

def calculate_log_probs(features, spam_word_probs, non_spam_word_probs, spam_prior):
                                                                             #compute log probability for spam
    log_spam_probs = np.dot(features, np.log(spam_word_probs)) + np.log(spam_prior)
                                                                             #compute log probability of non spam
    log_non_spam_probs = np.dot(features , np.log(non_spam_word_probs)) + np.log(1 - spam_prior)
    
    return log_spam_probs, log_non_spam_probs

def predict(features, spam_word_probs, non_spam_word_probs, spam_prior):
    log_spam_probs, log_non_spam_probs = calculate_log_probs(features, spam_word_probs, non_spam_word_probs, spam_prior)
                                                                            
    return np.where(log_spam_probs > log_non_spam_probs, 1, 0)              #if log_spam_probs > log_non_spam_prob, return 1
                                                                            #otherwise return 0

def train_naive_bayes(features, labels):                                     #Train Naive Bayes
    spam_prior= calculate_prior(labels)                                     
    spam_word_probs, non_spam_word_probs = calculate_word_probs(features, labels) 
    return spam_prior, spam_word_probs, non_spam_word_probs


data, labels= load_data('spam_or_not_spam.csv')                             #load and split data into 70-30
train_data, test_data, train_labels, test_labels= train_test_split(data, labels, test_size= 0.3, random_state= 9)

train_features= extract_features(train_data)                                 #extract feature for training
test_features= extract_features(test_data)                                   # for testing

                                                                             #train Naive Bayes Model
spam_prior, spam_word_probs, non_spam_word_probs = train_naive_bayes(train_features, train_labels)

# Predict and evaluate Naive Bayes accuracy
nb_predictions = predict(test_features, spam_word_probs, non_spam_word_probs, spam_prior)
nb_accuracy = np.mean(nb_predictions == test_labels)
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")


#Apply SVM to train
svm_classifier= SVC()                                                      #instance of SVC from sklearn.svm
svm_classifier.fit(train_features, train_labels)                           #fit method trains the data of training set

svm_predictions= svm_classifier.predict(test_features)                     #SVM model predicts labels for test features
svm_accuracy= np.mean(svm_predictions== test_labels)                       #SVM accuracy, returns true for correct predictions

print(f"SVM Accuracy:{svm_accuracy *100: .2f}%")  



#function to read emails from current directory

def read_emails_from_folder(folder_name= " test"):
    email_files= sorted(
        [i for i in os.listdir(folder_name) if i.endswith('.txt')],
        key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else float('inf')
    )                                                                         #get list of all files in .txt form in folder
    emails= []                                                                #initialize empty list to store email contents
    
    for j in email_files:                                                     #iterate through each file in the list of emails
        file_path= os.path.join(folder_name, j)                               #construct full path to the email file
        
        with open(file_path, 'r', encoding= 'utf-8') as file:                 #open the email file in read mode by utf encoding
            email_content= file.read().strip()                                #read the content of email
            emails.append((j, clean_text(email_content)))                     #append the cleaned content in the list
    return emails  

#Function to classify emails from the folder using Naive Bayes, SVM

def classify_emails_in_folder(folder_name, spam_word_probs, non_spam_word_probs, spam_prior, svm_classifier, output_file="email_predictions.csv"):
    emails = read_emails_from_folder(folder_name)
    j = [email[0] for email in emails]                                           # Extract file names
    email_contents = [email[1] for email in emails]                              # Extract email contents
    
    features = extract_features(email_contents)
    
                                                                                   # Naive Bayes predictions
    nb_predictions = predict(features, spam_word_probs, non_spam_word_probs, spam_prior)
    
                                                                                    # SVM predictions
    svm_predictions = svm_classifier.predict(features)
    
    
    print('Printing few predictions')
    for i in range(min(15, len(nb_predictions))):  
        
        print(f"{j[i]} - Naive Bayes: {'Spam(1)' if nb_predictions[i] == 1 else 'Non-Spam(0)'}, SVM: {'Spam(1)' if svm_predictions[i] == 1 else 'Non-Spam(0)'}")
    
                                                                                    # Save all predictions to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Naive Bayes Prediction", "SVM Prediction"])   # Header for the CSV
        for i in range(len(j)):
            writer.writerow([j[i], "Spam(1)" if nb_predictions[i] == 1 else "Non-Spam(0)", "Spam(1)" if svm_predictions[i] == 1 else "Non-Spam(0)"])
            
classify_emails_in_folder("test", spam_word_probs, non_spam_word_probs, spam_prior, svm_classifier)

        
    
    


# In[ ]:




