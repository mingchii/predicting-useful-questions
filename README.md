# predicting-useful-questions on Stack Overflow

 The dataset has 7468 observations, with each observation recording text data associated with both the title and the body 
 of the associated question and also the score of the question. 
 
 After the question is posted, it would be beneficial for Stack Overflow to get immediate sense of whether the question is useful or not, and promote them to the top of the page. In this project, I built models to predict whether a ggplot2 question is useful, based on the title and the body text data associated with each question. To be specific, a question is useful if its score is greater than or equal to one. 
 
 To begin with, I start by cleaning up the dataset using R package 'tm.plugin.webmining'. 
 1. the function 'extractHTMLStrip' is useful in removing html tags.
 2. cleaning up by removing punctuation, change to lower case, stopwords...etc.
 3. process title and body text separately then join them together.
 
After spliting training and testing datatset, we compare the outcome of five models such as baseline, logistic regression, CART, random forest, and boosting model. 
 
