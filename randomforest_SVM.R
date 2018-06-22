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

train = as.tibble(fread("train.csv",na.strings = c("-1","-1.0"))) #given train data
test = as.tibble(fread("test.csv",na.strings = c("-1","-1.0"))) #given test data


test$target = 0 #creating a target variable in test data
test$data = "test" #creating another variable to identify the test data rows
test = test[, c(1, 60, 59, 2:58)] #reforming with newly created variables

train$data = "train" #creating another variable to identify the train data rows
train = train[, c(1, 60, 2:59)] #reforming with newly created variables

combined_data = as.data.frame(rbind(train,test)) #combining test and train data
dim(combined_data) #dimensions of combined data

combined_data <- combined_data %>%
  mutate_at(vars(ends_with("cat")), funs(as.factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical)) %>%
  mutate(target = as.factor(target))

str(combined_data)

missing_values = as.data.frame(colSums(is.na(combined_data))) 
missing_values

vectordrop <- combined_data[, lapply( combined_data, 
                                      function(m) sum(is.na(m)) / length(m) ) >= .05 ]
#removing the columns in vectordrop from the main data
combined_data = combined_data[,!(colnames(combined_data) %in% colnames(vectordrop))]
dim(combined_data)


miss_pct <- sapply(combined_data, function(x) { sum(is.na(x)) / length(x) })
miss_pct <- miss_pct[miss_pct > 0]
names(miss_pct) #columns with missing data
#rm(miss_pct)

mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

df = combined_data #just making typing easy :P
rm(combined_data) #saving RAM
df$ps_ind_02_cat[is.na(df$ps_ind_02_cat)]<-mode(df$ps_ind_02_cat) #imputing with mode
df$ps_ind_04_cat[is.na(df$ps_ind_04_cat)]<-mode(df$ps_ind_04_cat)
df$ps_ind_05_cat[is.na(df$ps_ind_05_cat)]<-mode(df$ps_ind_05_cat)
df$ps_car_01_cat[is.na(df$ps_car_01_cat)]<-mode(df$ps_car_01_cat)
df$ps_car_02_cat[is.na(df$ps_car_02_cat)]<-mode(df$ps_car_02_cat)
df$ps_car_07_cat[is.na(df$ps_car_07_cat)]<-mode(df$ps_car_07_cat)
df$ps_car_09_cat[is.na(df$ps_car_09_cat)]<-mode(df$ps_car_09_cat)
df$ps_car_11[is.na(df$ps_car_11)]<-mode(df$ps_car_11)
df$ps_car_12[is.na(df$ps_car_12)]<-mean(df$ps_car_12,na.rm=T) #imputing with mean

sum(is.na(df))


TRAIN <- df[1:595212,-2] #train data set after pre-processing
dim(TRAIN)
TEST <- df[595213:1488028,-c(2,3)] #test data set after pre-processing
dim(TEST)


table(TRAIN$target)

table(TRAIN$target)/nrow(TRAIN)

balanced_train <- ovun.sample(target~.,data=TRAIN,method = "both",N = 90000,p=.5,seed=1)$data
head(balanced_train) #head of the dataset with balanced target
dim(balanced_train) #dimensions of the dataset with balanced target.

sum(is.na(balanced_train)) #checking for any Missing values
balanced_train <-as.data.frame(balanced_train)

table(balanced_train$target)

table(balanced_train$target)/nrow(balanced_train)

str(balanced_train)

s=sample(nrow(balanced_train),round(nrow(balanced_train)*0.7),replace=FALSE)
train = balanced_train[s,]
test = balanced_train[-s,]
dim(train);dim(test)


unique(train$ps_car_11_cat)
dim(train)
sapply(train,class)
train_rf = subset(train,select=-ps_car_11_cat)
#rm(train) #train without ps_car_11_cat
test_rf = subset(test,select=-ps_car_11_cat)

library("randomForest")
model_rf = randomForest(as.factor(target) ~. , data = train_rf) # Fit Random forest
saveRDS(model_rf, "mymodel.rds")
model_rf3<- readRDS("mymodel.rds")
summary(model_rf)
png("Model_RF.png")
plot(model_rf, main = "Model_RF") #Plot for number of trees and error
dev.off()

png("Var_imp.png")
varImpPlot(model_rf) #Plot for Important variable
dev.off()

pred_rf <- predict(model_rf3,test_rf) #predictin using the model
#pred_rf <- predict(model_rf3,TEST)
#write.csv(pred_rf,"results.csv")
summary(pred_rf)


confusionMatrix(pred_rf,test_rf$target)
library(pROC)
#ROC plot for the model
aucrf <- roc(as.numeric(test_rf$target), as.numeric(pred_rf),  ci=TRUE)
png("ROC.png")
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')
dev.off()
library(ROCR)
pr <- prediction(as.numeric(pred_rf),as.numeric(test_rf$target))
prf <- performance (pr,measure= "tpr", x.measure="fpr")
png("rf2_ROC.png")
plot(prf)
dev.off()


# Only for testing Naive Bayes

x_train = train
x_test = test
y_train = x_train$target
y_test = x_test$target
x_train$target = NULL
x_test$target = NULL
dim(x_train);dim(x_test)
length(y_train);length(y_test)

library(e1071)
library(caret)

x_cat = cbind(x_train,y_train)

#fit = naiveBayes(y_train ~., data = x_cat)
#pred = predict(fit,x_test)
#pred = predict(fit,x_test[,-1, drop=FALSE])
#pred = predict(fit, list(var = x_test[, -1]))

# SVM Model

fit = svm(y_train ~., data = x_cat)
saveRDS(fit, "svm.rds")
svm_model<- readRDS("svm.rds")
pred = predict(svm_model,x_test)
write.csv(pred,"svm_pred.csv")
pred=read.csv("svm_pred.csv")
#pred = predict(fit,x_test[,-1, drop=FALSE])
#pred = predict(fit, list(var = x_test[, -1]))
library(ROCR)
pr <- prediction(as.numeric(pred),as.numeric(test$target))
prf <- performance (pr,measure= "tpr", x.measure="fpr")

png("svm_ROC.png")
plot(prf)
dev.off()


summary(pred)
confusionMatrix(pred,y_test)

aucrf <- roc(as.numeric(train$target), as.numeric(pred),  ci=TRUE)
png("ROC2.png")
