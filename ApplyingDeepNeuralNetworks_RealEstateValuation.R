#Libraries
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)

#Reading data
data <- read.csv("~/LearningR/RealEstateValuation.csv", header = TRUE)
str(data)


# Let us first Convert all factor variables into numeric variables as DNN requires numeric variables.

data %<>% mutate_if(is.factor,as.numeric)

# Converting the only integer variable into numeric variable
data$X4 <- as.numeric(data$X4)
str(data)

#Neural Network Visualization
# There are two hidden layers with 10 and 5 neurons, respectively.
# The input layer has one neuron for each independent variable (5 variables). 
# Output layer has one node for the response variable, Y.

n <- neuralnet(Y~., data = data, hidden = c(10,5), linear.output = F, lifesign = 'full', rep = 1)
plot(n)
plot(n, col.entry = 'darkgreen', col.hidden.synapse = 'darkgreen', show.weights = F, information = F, fill = 'lightblue')

# Creating a Matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Data Partitioning
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))
training <- data[ind==1,1:5]
test <- data[ind==2, 1:5]
trainingtarget <- data[ind==1, 6]
testtarget <- data[ind==2, 6]


# We will do the zscore Normalization by subtracting from mean and dividing by standard deviation.
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
#We use the mean and sd of the training data.
test <- scale(test, center = m, scale = s)    

# Create Model
# One hidden layer with 5 neurons. 
# For activation function we are going to use Rectified Linear Unit(relu) in the hidden layers. 
# Input layer will 5 neurons as there are 5 features or independent variables.
# Finally we will have oe neuron for the output layer, Y. 
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 5, activation = 'relu', input_shape = c(5)) %>%
  layer_dense(units = 1)


# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

# Evaluate the Model
model %>% evaluate(test, testtarget)

#Predictions
pred <- model %>% predict(test)
mean((testtarget-pred)^2) # Also gives us the same error value. 
plot(testtarget, pred)


# Fine tune Model
# Now we will add another hidden layer.
model <- keras_model_sequential()
model %>%
  layer_dense(units = 10, activation = 'relu', input_shape = c(5)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 1)
summary(model) 

#Compile.
model %>% compile(loss = 'mse', optimizer = 'rmsprop', metrics = 'mae')

#Fit model
mymodel <- model %>% 
  fit(training,trainingtarget, epochs = 100, batch_size = 32, validation_split = 0.2)

#Evaluate Model with Test data
model %>%   evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget -pred)^2) 
plot(testtarget, pred)


#More changes
# We increase the number of neurons in the first hidden layer to 100.
# Also add adrop out layers.It helps avoid overfitting. 
# It means for the first hidden layer during the training 40% of the neurons are dropped to zero.
model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(5)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)
summary(model)

#Compile.
model %>% compile(loss = 'mse', optimizer = optimizer_rmsprop(lr = 0.005), metrics = 'mae')

#Fit model
mymodel <- model %>% 
  fit(training,trainingtarget, epochs = 100, batch_size = 32, validation_split = 0.2)

#Evaluate Model with Test data
model %>%   evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget -pred)^2) 
plot(testtarget, pred)

## The last model is an improvement with the least error.