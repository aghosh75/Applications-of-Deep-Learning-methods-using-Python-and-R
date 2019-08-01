# Needed libraries
library(maximin)

#Reading data
getwd()
data <- read.csv("~/LearningR/admissions.csv", header = TRUE)
str(data)
View(data)
hist(data$gre)

# We have to do a min-max normalization to ensure the variables are between 0 and 1 before we apply Neural Networks.
# We subtract each feaure fro its minimum value an dthen divide by the difference between the maximum and minimum values.
data$gre <- (data$gre - min(data$gre))/(max(data$gre) - min(data$gre))
data$gpa <- (data$gpa - min(data$gpa))/(max(data$gpa) - min(data$gpa))
data$rank <- (data$rank - min(data$rank))/(max(data$rank)-min(data$rank)) 


# Partitioning the data into training and test datasets.
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training <- data[ind==1,]
testing <- data[ind==2,]


# building the Neural Network.
library(neuralnet)
set.seed(333)
# We store the neural network in n.
# We will keep only one hidden ntwork to start with
# err.fct Is a differentiable function used for the calculation of the error.Here we use cross entropy.
n <- neuralnet(admit~gre+gpa+rank,
               data = training,
               hidden = 1,
               err.fct = "ce",
               linear.output = FALSE)
plot(n)


# Makking Predictions. 
# for making predictions using neural networks the feature or target variable should be excluded. 
output <- compute(n, training[,-1])
head(output$net.result)
# We can compare the predicted values above with the training data.
head(training[1,])
# So, the first student is in a way misclassifed.

# Node Output Calculations with Sigmoid Activation Function
in4 <- 0.0455 + (0.82344*0.7586206897) + (1.35186*0.8103448276) + (-0.87435*0.6666666667)
out4 <- 1/(1+exp(-in4))
in5 <- 1.69989 +(-13.80226*out4)
out5 <- 1/(1+exp(-in5))

# Confusion Matrix & Misclassification Error - training data
output <- compute(n, training[,-1])
p1 <- output$net.result
pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(pred1, training$admit)
tab1
#Misclassification Error
1-sum(diag(tab1))/sum(tab1)

# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2, testing$admit)
tab2
1-sum(diag(tab2))/sum(tab2)


#More Neurons in hidden layer - 5 hidden layers
set.seed(333)
n <- neuralnet(admit~.,data = training, hidden = 5, err.fct = 'ce', linear.output = FALSE)
plot(n)

# Now we repeat the confusion marix for traiing and testing.


#Neural network with 2 hidden layers
set.seed(333)
n <- neuralnet(admit~.,data = training, hidden = c(2,1), err.fct = 'ce', linear.output = FALSE)
plot(n)

# Again we check for the performance of the training and test data set.


#Neural network Repeat Calculations
set.seed(333)
#rep is the number of repitions of the neural network model.
n <- neuralnet(admit~.,data = training, hidden = 5, err.fct = 'ce', linear.output = FALSE, lifesign = 'full', rep = 5)
# 4 out of the 5 repititions converged. the first one had the minimal error. 
# We ask to plot the fist neural network
plot(n, rep= 1)

# We specify rep =1 before we get accuracy and confusion matrix. 
output <- compute(n, training[, -1], rep = 1)
p1 <- output$net.result
pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(pred1, training$BankingCrisisdummy)
tab1
1- sum(diag(tab1))/sum(tab1)

output <- compute(n, test[, -1], rep = 1)
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2, test$BankingCrisisdummy)
tab2
1- sum(diag(tab2))/sum(tab2) 


#We can also specify which algorithm we want. Deafult is RPROP (resilient backpropagation with weight backtacking)
# We can also specify the maximum steps for the algorithm.
set.seed(333)
n <- neuralnet(admit~.,data = training, hidden = 5, err.fct = 'ce', linear.output = FALSE, lifesign = 'full', rep = 5, algorithm = "rprop+", stepmax = 100000)
plot(n)
