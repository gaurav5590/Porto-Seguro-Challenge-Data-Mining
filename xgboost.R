library(dplyr) #data manipulation
library(readr) #input/output
library(data.table) #data manipulation
library(stringr) #string manipulation
library(caret)  #model evaluation (confusion matrix)
library(tibble) #data wrangling
library("ROSE") #over/under sampling
library("randomForest") #random forest model building
library(pROC) #ROC plots
library("MLmetrics") #Normalized Gini
library(xgboost)
library(verification)

train = as.tibble(fread("train.csv",na.strings = c("-1","-1.0"))) #given train data
test = as.tibble(fread("test.csv",na.strings = c("-1","-1.0"))) #given test data

dim(train);dim(test) #dimensions of train and test
table(train$target) #examining the target variable

test$target = 0 #creating a target variable in test data
test$data = "test" #creating another variable to identify the test data rows
test = test[, c(1, 60, 59, 2:58)] #reforming with newly created variable

train$data = "train" #creating another variable to identify the train data rows
train = train[, c(1, 60, 2:59)] #reforming with newly created variables

combined_data = as.data.frame(rbind(train,test)) #combining test and train data
dim(combined_data) #dimensions of combined data


combined_data <- combined_data %>%
  mutate_at(vars(ends_with("cat")), funs(as.factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical)) %>%
  mutate(target = as.factor(target))

missing_values = as.data.frame(colSums(is.na(combined_data))) 
missing_values
rm(missing_values) #viewing and removing missing values

vectordrop <- combined_data[, lapply( combined_data, 
                                      function(m) sum(is.na(m)) / length(m) ) >= .05 ]

colnames(vectordrop)
#removing the columns in vectordrop from the main data
combined_data = combined_data[,!(colnames(combined_data) %in% colnames(vectordrop))]
dim(combined_data) 
#rm(vectordrop) 
combined_data1 <- bind_rows(train,test)
combined_data$target <- combined_data1$target
dim(combined_data)
#rm(combined_data1)
#Now lets impute remaining columns with missing data

miss_pct <- sapply(combined_data, function(x) { sum(is.na(x)) / length(x) })
miss_pct

miss_pct <- miss_pct[miss_pct > 0]
names(miss_pct) #columns with missing data

mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

df = combined_data #just making typing easy :P
#rm(combined_data) #saving RAM
df$ps_ind_02_cat[is.na(df$ps_ind_02_cat)]<-mode(df$ps_ind_02_cat) #imputing with mode
df$ps_ind_04_cat[is.na(df$ps_ind_04_cat)]<-mode(df$ps_ind_04_cat)
df$ps_ind_05_cat[is.na(df$ps_ind_05_cat)]<-mode(df$ps_ind_05_cat)
df$ps_car_01_cat[is.na(df$ps_car_01_cat)]<-mode(df$ps_car_01_cat)
df$ps_car_02_cat[is.na(df$ps_car_02_cat)]<-mode(df$ps_car_02_cat)
df$ps_car_07_cat[is.na(df$ps_car_07_cat)]<-mode(df$ps_car_07_cat)
df$ps_car_09_cat[is.na(df$ps_car_09_cat)]<-mode(df$ps_car_09_cat)
df$ps_car_11[is.na(df$ps_car_11)]<-mode(df$ps_car_11)
df$ps_car_12[is.na(df$ps_car_12)]<-mean(df$ps_car_12,na.rm=T) #imputing with mean
sapply(df,class)
#checking for missing values
sum(is.na(df))

TRAIN <- df[1:595212,-2] #train data set after pre-processing
dim(TRAIN)
TEST <- df[595213:1488028,-c(2,3)] #test data set after pre-processing
dim(TEST)

balanced_train <- ovun.sample(target~.,data=TRAIN,method = "both",N = 90000,p=.5,seed=1)$data
balanced_train <- as.data.frame(model.matrix(~. - 1, data = balanced_train))
s=sample(nrow(balanced_train),round(nrow(balanced_train)*0.7),replace=FALSE)
train = balanced_train[s,]
test = balanced_train[-s,]
dim(train);dim(test)


sapply(train,class)
x_train <- train
y_train <- as.factor(x_train$target)
x_train$target <- NULL
x_test <- test
y_test <- as.factor(x_test$target)
x_test$target <- NULL

levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")


normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

giniSummary <- function (data, lev = "Yes", model = NULL) {
  levels(data$obs) <- c('0', '1')
  out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
  names(out) <- "NormalizedGini"
  out
}

trControl = trainControl(
  method = 'cv',
  number = 2,
  summaryFunction = giniSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)


tuneGridXGB <- expand.grid(
  nrounds=c(350),
  max_depth = c(4, 6),
  eta = c(0.05, 0.1),
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0))

xgbmod <- train(
  x = x_train,
  y = y_train,
  method = 'xgbTree',
  metric = 'NormalizedGini',
  trControl = trControl,
  tuneGrid = tuneGridXGB)

saveRDS(xgbmod, "xgboost.rds")
xg_boost<- readRDS("xgboost.rds")
preds <- predict(xg_boost, newdata = x_test)
confusionMatrix(preds, y_test)

library(ROCR)
pr <- prediction(as.numeric(preds),as.numeric(test$target))
prf <- performance (pr,measure= "tpr", x.measure="fpr")
png("xgROC.png")
plot(prf)
dev.off()

varImp(xgbmod)
preds_prob <- predict(xgbmod, newdata = x_test, type = "prob")
y_test_prob <- y_test
levels(y_test_prob) <- c("0", "1")
y_test_raw <- as.numeric(levels(y_test_prob))[y_test_prob]


normalizedGini(y_test_raw, preds_prob$Yes)
png("xgboost_model.png")
plot(xgbmod)
dev.off()

png("roc_plot_xgboost.png")
roc.plot(y_test_raw, preds_prob$Yes)
dev.off()
