
library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tm.plugin.webmining)
library(data.table)
library(boot)


tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

# strwrap(questionbody[[1]])



# StackOverflow <-  read.csv("ggplot2questions2016_17.csv", stringsAsFactors=FALSE)

StackOverflow <- fread("ggplot2questions2016_17.csv", stringsAsFactors=FALSE)

# convert score to usefulness
StackOverflow$usefulness <- as.factor(as.numeric(StackOverflow$Score >= 1))
StackOverflow$Score <- NULL

##### processing body

# remove HTML Tag
questionbody <-  Corpus(VectorSource(StackOverflow$Body))
Clean <- function(HTML){
  return(gsub("<.*?>", "", HTML))

}
for  (i in 1:length(questionbody)) { questionbody[[i]] <- Clean(questionbody[[i]]) }

# remove punctuation
questionbody <-  tm_map(questionbody, removePunctuation)

# covert to lower case
questionbody <-  tm_map(questionbody, tolower)

# remover stop words
questionbody <-  tm_map(questionbody, removeWords, stopwords("english"))

# stemming
questionbody <-  tm_map(questionbody, stemDocument)

# remove ggplot and numbers and symbol x,y,z,q
for (i in 1:length(questionbody)) {questionbody[[i]] <- gsub(c("ggplot"), "", questionbody[[i]])}
for (i in 1:length(questionbody)) { for (j in 0:9) {questionbody[[i]] <- gsub(as.character(j), "", questionbody[[i]]) } }
questionbody <- tm_map(questionbody, removeWords, c("x","y","z","q"))

# examine frequencies and remove low frequency words in body and I keep the words with at least 0.05% of frequecy
freqbody <-  DocumentTermMatrix(questionbody)
freqbody
sparsebody <- removeSparseTerms(freqbody, 0.86)


length(findFreqTerms(freqbody, lowfreq = 50))
length(findFreqTerms(freqbody, lowfreq = 20))
sparsebody <- removeSparseTerms(freqbody, 0.99)
sparsebody
sparsebody <- removeSparseTerms(freqbody, 0.995)
sparsebody
sparsebody <- removeSparseTerms(freqbody, 0.86)

# create the clean independent variable of body text as data.frame
finalbody <- as.data.frame(as.matrix(sparsebody))
colnames(finalbody) <- make.names(colnames(finalbody))
colnames(finalbody) <- paste("body", colnames(finalbody), sep = ".")



##### processing title

questiontitle <-  Corpus(VectorSource(StackOverflow$Title))

# remove punctuation
questiontitle <-  tm_map(questiontitle, removePunctuation)

# covert to lower case
questiontitle <- tm_map(questiontitle, tolower)

# remover stop words
questiontitle <-  tm_map(questiontitle, removeWords, stopwords("english"))

# stemming
questiontitle <-  tm_map(questiontitle, stemDocument)
strwrap(questiontitle[[1]])

# remove ggplot 
for (i in 1:length(questiontitle)) {questiontitle[[i]] <- gsub(c("ggplot"), "", questiontitle[[i]])}


# examine frequencies and remove low frequency words in title and I keep the words with at least 0.05% of frequecy
freqtitle <-  DocumentTermMatrix(questiontitle)
freqtitle
sparsetitle <- removeSparseTerms(freqtitle, 0.95)

sparsetitle <- removeSparseTerms(freqtitle, 0.99)
sparsetitle
sparsetitle <- removeSparseTerms(freqtitle, 0.995)
sparsetitle
sparsetitle <- removeSparseTerms(freqtitle, 0.95)

# create the clean independent variable of title text as data.frame
finaltitle <- as.data.frame(as.matrix(sparsetitle))
colnames(finaltitle) <- make.names(colnames(finaltitle))
colnames(finaltitle) <- paste("title", colnames(finaltitle), sep = ".")


# the clean data set can be used to trainning and testing
clean.data <- as.data.frame(cbind(StackOverflow$usefulness,finaltitle, finalbody ))
colnames(clean.data)[1] <- "usefulness"

# splitting the data into training and testing set
set.seed(123)
split <- sample.split(clean.data$usefulness, SplitRatio = 0.7)
train <- filter(clean.data, split == TRUE)
test <- filter(clean.data, split == FALSE)
# bootstrap.test <- as.data.frame(matrix())
# for (k in 1:100) {bootstrap.test[k] <- sample(test, size = length(test$usefulness), replace = TRUE) }


# Base line
table(train$usefulness)
table(test$usefulness)
1-mean(as.numeric(test$usefulness)-1) # the base line accuracy of choosing not useful


# Logistic model
log.model <- glm(usefulness ~ . , data = train, family = "binomial")
summary(log.model)
log.predict <- predict(log.model, newdata = test, type = "response")
table(test$usefulness, log.predict > 0.5)
tableAccuracy(test$usefulness, log.predict > 0.5)
# training set R^2 
log.predict.train <- predict(log.model, type = "response")
table(train$usefulness, log.predict.train > 0.5)
tableAccuracy(train$usefulness, log.predict.train > 0.5)


# Stepwise regression
log.step.model <- step(log.model, direction = "backward")
summary(log.step.model)
length(log.step.model$coefficients)
log.step.predict <- predict(log.step.model, newdata = test, type = "response")
table(test$usefulness, log.step.predict > 0.5)
tableAccuracy(test$usefulness, log.step.predict > 0.5)
# training set R^2 
log.step.predict.train <- predict(log.step.model, type = "response")
table(train$usefulness, log.step.predict.train > 0.5)
tableAccuracy(train$usefulness, log.step.predict.train > 0.5)




# Cart model
set.seed(123)
train.cart <- train(usefulness ~ ., 
                    data = train, 
                    method = "rpart", 
                    tuneGrid = data.frame(cp = seq(0,0.2, 0.002)), 
                    trControl = trainControl(method = "cv", number = 5), 
                    metric = "Accuracy")
train.cart$results

ggplot(train.cart$results, aes(x = cp, y = Accuracy)) + 
  geom_point(size = 2) + 
  geom_line() + 
  ylab("CV Accuracy") + 
  theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

cart.mod <- train.cart$finalModel
prp(cart.mod)

cart.predict <- predict(cart.mod, newdata = test, type = "class")
table(test$usefulness, cart.predict)
tableAccuracy(test$usefulness, cart.predict)


# Random Forest model
set.seed(123)
train.rf <-  train(usefulness ~ ., 
                   data = train, 
                   method = "rf", 
                   tuneGrid = data.frame(mtry = 1:30), 
                   trControl = trainControl(method = "cv", number = 5),
                   metric = "Accuracy")
 
train.rf$results

ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 3) + geom_line() + 
  ylab("Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

rf.model  <-  train.rf$finalModel
rf.predict <-  predict(rf.model, newdata = test)
table(test$usefulness, rf.predict)
tableAccuracy(test$usefulness, rf.predict)


# Boosting model
tune.grid <- expand.grid(n.trees = seq(75, 125, by = 5)*50, interaction.depth = seq(2, 10, by = 2),
                    shrinkage = 0.001, n.minobsinnode = 10)
set.seed(123)
train.boost <- train(usefulness ~ ., 
                     data = train, 
                     method = "gbm", 
                     tuneGrid = tune.grid, 
                     trControl = trainControl(method = "cv", number = 5),
                     metric = "Accuracy", 
                     distribution = "bernoulli")

train.boost$results

ggplot(train.boost$results, aes(x = n.trees, y = Accuracy, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

boost.model <- train.boost$finalModel
test.mm <- model.matrix(usefulness ~ . +0, data = test) %>% as.data.frame()

boost.predict <- predict(boost.model, newdata = test.mm, n.tree = 4000, type = "response")
table(test$usefulness, boost.predict < 0.5)
tableAccuracy(test$usefulness, boost.predict < 0.5)




## Bootstrap 

allmetrics <- function(data, index) {  
  t <- data$test[index]
  p <- data$pred[index]
  TN <- sum(t == 0 & p == 0)
  FP <- sum(t == 0 & p == 1)
  FN <- sum(t == 1 & p == 0)
  TP <- sum(t == 1 & p == 1)
  
  Accuracy <- (TP+TN)/(TN+FP+FN+TP)
  TPR <- TP/(TP+FN)
  FPR <- FP/(FP+TN)
  
  return(c(Accuracy, TPR, FPR))
  }

bootstrap.data <- data.frame(test = test$usefulness, pred = rf.predict)

set.seed(123)
finalmodel.boot <-  boot(bootstrap.data, allmetrics, R = 10000)
finalmodel.boot

# confidence intervals  
boot.ci(finalmodel.boot, index = 1, type = "basic") # Accuracy
boot.ci(finalmodel.boot, index = 2, type = "basic") # TPR
boot.ci(finalmodel.boot, index = 3, type = "basic") # FPR








#### retrain for the FPR

# Cart model
set.seed(123)
train.fpr.cart <- train(usefulness ~ ., 
                    data = train, 
                    method = "rpart", 
                    tuneGrid = data.frame(cp = seq(0,0.2, 0.002)), 
                    trControl = trainControl(method = "cv", number = 5), 
                    metric = "FPR")
train.fpr.cart$results

ggplot(train.fpr.cart$results, aes(x = cp, y = Accuracy)) + 
  geom_point(size = 2) + 
  geom_line() + 
  ylab("CV FPR") + 
  theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

cart.fpr.mod <- train.fpr.cart$finalModel
prp(cart.fpr.mod)

cart.fpr.predict <- predict(cart.fpr.mod, newdata = test, type = "class")
table(test$usefulness, cart.fpr.predict)
tableAccuracy(test$usefulness, cart.fpr.predict)


# Random Forest model
set.seed(123)
train.rf <-  train(usefulness ~ ., 
                   data = train, 
                   method = "rf", 
                   tuneGrid = data.frame(mtry = 1:30), 
                   trControl = trainControl(method = "cv", number = 5),
                   metric = "FPR")

train.rf$results

ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 3) + geom_line() + 
  ylab("Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

rf.model  <-  train.rf$finalModel
rf.predict <-  predict(rf.model, newdata = test)
table(test$usefulness, rf.predict)
tableAccuracy(test$usefulness, rf.predict)



# Boosting model
tune.grid <- expand.grid(n.trees = seq(75, 125, by = 5)*50, interaction.depth = seq(2, 10, by = 2),
                         shrinkage = 0.001, n.minobsinnode = 10)
set.seed(123)
train.boost <- train(usefulness ~ ., 
                     data = train, 
                     method = "gbm", 
                     tuneGrid = tune.grid, 
                     trControl = trainControl(method = "cv", number = 5),
                     metric = "FPR", 
                     distribution = "bernoulli")

train.boost$results

ggplot(train.boost$results, aes(x = n.trees, y = Accuracy, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

boost.model <- train.boost$finalModel
test.mm <- model.matrix(usefulness ~ . +0, data = test) %>% as.data.frame()

boost.predict <- predict(boost.model, newdata = test.mm, n.tree = 4000, type = "response")
table(test$usefulness, boost.predict < 0.5)
tableAccuracy(test$usefulness, boost.predict < 0.5)



