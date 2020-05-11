# Import Data #
movie <- read.csv("E:/My Dictionary/Using R/Data/Movie_classification.csv")
View(movie)
str(movie)
movie$Start_Tech_Oscar <- as.factor(movie$Start_Tech_Oscar)

# Data Preprocessing #
summary(movie) #there are missing values in variable Time_taken
movie$Time_taken[is.na(movie$Time_taken)] <- mean(movie$Time_taken,na.rm = TRUE) #imputasi with mean because it is numerical variable

# Test-Train Split
install.packages('caTools')
library(caTools)
set.seed(1)
split <- sample.split(movie,SplitRatio = 0.8)
traine <- subset(movie,split == TRUE)
teste <- subset(movie,split == FALSE)

############################### MODELING #################################
install.packages("xgboost") 
library(xgboost)

#the categorical variable must be changed to dummy variable in this method, so we need to change them first:
trainY <- traine$Start_Tech_Oscar == "1" #change response variable with "TRUE" and "FALSE"
head(trainY)
trainX <- model.matrix(Start_Tech_Oscar~.-1,data=traine) #All categorical variable will be changed to dummy variable and "-1" means that the first dummy variable will be removed because n-1
head(trainX)
trainX <- trainX[,-12] #remove first dummy variable of variable X3D_available (delete additional variable)

testY = teste$Start_Tech_Oscar == "1"
testX <- model.matrix(Start_Tech_Oscar ~ .-1, data = teste)
testX <- testX[,-12]
head(testX)

Xmatrix <- xgb.DMatrix(data = trainX, label= trainY) 
Xmatrix_t <- xgb.DMatrix(data = testX, label = testY)
Xgboosting <- xgboost(data = Xmatrix, # the data   
                      nround = 50, # max number of boosting iterations
                      objective = "multi:softmax",
                      eta = 0.3, 
                      num_class = 2, #because we have 2 categorical predictor variables 
                      max_depth = 10)

xgpred <- predict(Xgboosting, Xmatrix_t)
cm <- table(testY, xgpred)
cm
Accuracy <- (cm[1,1]+cm[2,2])/sum(cm)
Accuracy
