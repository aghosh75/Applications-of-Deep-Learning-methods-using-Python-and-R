library(keras)
library(tensorflow)
install_keras()
install_tensorflow()

#Reading data
data <- read.csv("~/LearningR/CTG.csv", header = TRUE)
str(data)


#Change to matrix
data<- as.matrix(data)
# We are going to remove default names of data bu using NULL.
dimnames(data) <- NULL


# Normalizatio of the data
data[, 1:21] <- normalize(data[, 1:21])
data[,22] <- as.numeric(data[,22]) -1
summary(data)


# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))

# For both traiing and test data we are using the first 21 columns.
training <- data[ind==1, 1:21]
test <- data[ind==1, 1:21]

# For target or dependent variable we will call it train and test target.
trainingtarget <- data[ind==1, 22]
testtarget <- data[ind==2, 22]


# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels)


# Next we will create te sequential model with 8 nodes in the hidden layer.
# For input shape we use 21 as we have 21 features or independent variables. 
# We have three categories in the output variable so we use 3 for the second dense layer.
model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(21))%>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)


# Now we Compile the model. 
# With the outcome variabe (NSP) having three categories we use categorical_crossentropy. 
# If it had two categories we will use binary_crossentropy.

model%>%
  compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')

#Fit Model Multiplayer perceptron neural network for multi-class Softmax Classification)
# Batch size is the numebr of samples we can use per gradient.
# We will store it in history.
history <- model %>% 
  fit(training, trainLabels, epoch = 200, batchsize = 32, validation_split = 0.2)
plot(history) 

## Training loss and validation loss should fall together. 
## The last iteration 1218 comes from 80% of the training data, 1523. 

#Evaluate Model with Test data
model1 <- model %>%
  evaluate(test, testLabels)

#Prediction and Confusion matrix
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)

# Next we make a table for the confusion matrix
table1<-table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

#Fine Tune Model

#Model 2:
model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(21))%>%
  layer_dense(units = 3, activation = 'softmax')

model2 <- model %>%
  evaluate(test, testLabels)

table2<-table(Predicted = pred, Actual = testtarget)


#Adding another layer to fine tune the model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(21))%>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)

model3 <- model %>%
  evaluate(test, testLabels)

table3<-table(Predicted = pred, Actual = testtarget)

# Comparing the three models
table1 
model1
table2
model2
table3
model3
