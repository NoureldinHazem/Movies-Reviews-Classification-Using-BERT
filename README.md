# Movies-Reviews-Classification-Using-BERT
</br>
Problem Statement:</br>
IMDB is the most globally famous movie reviews website where you can publish a review for
any film you watched. Classifying the positive reviews and the negative ones can be useful for
several purposes such as giving an overall rating for the film or making statistical analysis about
the preferences of people from different countries, age levels, etc... So IMDB dataset is released
which composed of 50k reviews labeled as positive or negative to enable training movie reviews
classifiers. Moreover, NLP tasks are currently solved based on pretrained language models such
as BERT. These models provide a deep understanding of both semantic and contextual aspects
of language words, sentences or even large paragraphs due to their training on huge corpus for
very long time. In this assignment you will download the IMDB dataset from kaggle using this
Link. Then, you will train BERT based classifier for movie reviews.</br></br>

Link of data:</br>
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews</br></br>

1. Data Split:</br>
Split your dataset randomly so that the training set would form 70% of the dataset, the
validation set would form 10% and the testing set would form 20% of it. You should keep
all the splits balanced. </br></br>

2. Text Pre-processing: </br>
Text pre-processing is essential for NLP tasks. So, you will apply the following steps on
our data before used for classification:
1) Remove punctuation.
2) Remove stop words.
3) Lowercase all characters.
4) Lemmatization of words.
</br></br>

3. Classification using BERT:</br>
You need to build a classifier model based on BERT. You can use transformers library
supplied by hugging face to get a pretrained and ready version of BERT model. It will
also help you to tokenize the input sentence in the BERT required form and to pad the
short sentences or trim the long ones. We will use the CLS token embedding outputs of
BERT as input to the hidden dense classification layers we need to add after BERT. This
embedding is of size 768.</br>
You need to add 4 hidden layers of 512, 256, 128, 64 units respectively before the output
layer. You will use binary cross entropy loss and adam optimizer.</br></br>

4. Validation and Hyperparameter Tuning:</br>
Use the validation split to evaluate the model performance after each training epoch then
save the model checkpoint to choose the one with the best performance as the final model.
You can use dropout between dense layers to avoid overfitting if it arises.</br>
Also, you need to tune the learning rate hyperparameter of Adam optimizer using the
performance on the validation set.</br></br>

5. Checking Pre-processing Importance:</br>
BERT is assumed to capture the semantic and contextual aspects of the language. So,
sometimes it is better to input the text to it without pre-processing. To check the preprocessing
importance on our task we will train the model twice one using the preprocessed
version of data and the other using the original version then test both models
using the testing set and compare between the results.</br>
Note that you need to repeat the validation and hyperparameter tuning steps in both
cases. Also, note that the model trained on pre-processed data must be validated and
tested using pre-processed data and vice versa.</br></br>

6. Report Requirements:</br>
You should report graphs representing the change of training and validation accuracies
with the number of training epochs for your experiments.</br>
You should report a graph comparing between the best validation accuracies for the
different values of learning rate.</br>
You should report the model accuracy, precision, recall, specificity and F-score as
well as the resultant confusion matrix using the testing set for the best model with
pre-processing and without.</br>
Your comments on all results and comparisons.</br></br>

7. Bonus</br>
Finetune the number of hidden dense layers we need to add for classification and the
number of units in each layer using the validation set. Then test the best model using
the testing set and report all the above required metrics.
