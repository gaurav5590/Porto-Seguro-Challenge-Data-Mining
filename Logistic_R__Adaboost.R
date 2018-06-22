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

combined_data = as.data.frame(rbind(train,test)) #combining test and train data
dim(combined_data) #dimensions of combined data



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

str(df)

df$ps_car_11 <- as.numeric(df$ps_car_11)

for (i in 1:ncol(df)){
  df[,i] = as.numeric(df[,i])
}

table(df$target)


TRAIN <- df[1:595212,-1] #train data set after pre-processing
dim(TRAIN)
TEST <- df[595213:1488028,-c(1,2)] #test data set after pre-processing
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

balanced_train$target = as.numeric(balanced_train$target)

s=sample(nrow(balanced_train),round(nrow(balanced_train)*0.7),replace=FALSE)
train_ = balanced_train[s,]
test_ = balanced_train[-s,]
dim(train_);dim(test_)





########## LOGISTIC REGRESSION


model <- glm(target ~.,family=binomial(link='logit'),data=train_)


summary(model)


fitted.results <- predict(model,newdata=subset(test_,select=c(2:54)),type='response')
fitted.results_new <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results_new != test_$target)
print(paste('Accuracy',1-misClasificError))

table(test_$target, fitted.results_new,dnn=c("Actual","Predicted"))


#confusionMatrix(data = fitted.results, reference = test$target)


library(ROCR)

p <- predict(model, newdata=subset(test_,select=c(2:54)), type="response")
pr <- prediction(p, test_$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

png("Logistic_roc.png")

plot(prf)

dev.off()




############## ADABOOST
library(ada)
gen1 <- ada(target~.,train_ ,test_[,-1], test_[,1],"exponential", type = "gentle", iter = 70,na.action=na.rpart)

summary(gen1)
