# Data-Science-Challenge-HM-Submission

The python script is only for implementation purposes. A jupyter notebook with the same functionality but a better visualisation is included in the repository. Written description and somewhere graphical representations are also provided in the jupyter notebook for a better explanation. 


Steps to execute the python script:- 

Clone the repository by using the command "git clone https://github.com/yash-td/Data-Science-Challenge-HM-Submission.git".
Follow these commands to run the script: 

1) Open the terminal or command prompt and type "cd Data-Science-Challenge-HM-Submission" and press enter.

2) Before running the script, we need to install all the required python libraries and packages, hence we first 
install the same using the following command "pip install -r requirements.txt" (use pip3 for MacOS)

3) Simply run the script using the line of code below:

MacOS: python3 main.py

Windows: python main.py

4) After about 25-30 mins of running, a csv file named 'results.csv' will be generated with the text classification accuracy scores for 5 different models. 

5) After about 20mins of running a csv file named 'results.csv' will be generated with the text classification accuracy scores for 6 different models. 


--------------------------------------------------------------------------------------------------------------------------

Discussion:- 

This was a task of multilabel classification where a classifier model was trained with 9000 cases and tested on 1000 cases. The total number of unique labels (aticle violations) were 10 (0,1,2,3,4,5,6,7,8,9). For each case there were one, none or more than one violations. Hence I used a multi label binarizer so that the classifiers could interpret the multiple labels for each cases. To extract the features I used a TFIDF transformer and a vectorizer which considers the inverse document frequencey of all the words from the entire corpus. Five different classification models were fitted to our data and further a grid search cv was performed for the model with the best accuracy to further improve the performance.

It may seem that the accuracies of individual class prediction is very high but for the entire classifier is low. This is because there are a lot of cases where more than one article is violated. Hence for the classifier to count the entire instance as True Positive all the violations need to match. The indivual accuries from the confusion matrix show that how well the violations of individual class is predicted. For example:- The classifier predicts if the article 2 is violated or not with a 93% accuracy and a f1 score of 0.87. The overall accuracy of the classifier is 56% with a weighted average f1 score of 0.69. 

