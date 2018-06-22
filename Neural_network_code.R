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

combined_data = as.data.frame(rbind(train,test))

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

table(df$target)

for (i in 1:ncol(df)){
  df[,i] = as.numeric(df[,i])
}

TRAIN <- df[1:595212,-1] #train data set after pre-processing
dim(TRAIN)
TEST <- df[595213:1488028,-c(1,2)] #test data set after pre-processing
dim(TEST)


balanced_train <- ovun.sample(target~.,data=TRAIN,method = "both",N = 90000,p=.5,seed=1)$data
head(balanced_train) #head of the dataset with balanced target
dim(balanced_train) #dimensions of the dataset with balanced target.

sum(is.na(balanced_train)) #checking for any Missing values
balanced_train <-as.data.frame(balanced_train)

table(balanced_train$target)

str(balanced_train)

balanced_train$ps_car_11 <- NULL
balanced_train$ps_car_12 <- NULL
balanced_train$ps_car_14 <- NULL

balanced_train$ps_ind_02_cat <- NULL
balanced_train$ps_ind_04_cat <- NULL
balanced_train$ps_ind_05_cat <- NULL

balanced_train$ps_car_01_cat <- NULL
balanced_train$ps_car_02_cat <- NULL
balanced_train$ps_car_07_cat <- NULL
balanced_train$ps_car_09_cat <- NULL


for(i in 1:ncol(balanced_train)){
  x <- balanced_train[,i]
  balanced_train[,i] <- scale(x, min(x), max(x)-min(x))
}

index <- sample(1:nrow(balanced_train),round(0.75*nrow(balanced_train)))

train_ <- balanced_train[index,]
test_ <- balanced_train[-index,]

library(neuralnet)
n <- names(train_)
f <- as.formula(paste("target ~", paste(n[!n %in% "target"], collapse = " + ")))






nn <- neuralnet(f, data=train_, hidden=15,threshold = 0.05, stepmax=1e+06,linear.output=F)



saveRDS(nn,"nn.rds")

nn_tem <- readRDS("nn.rds") 

png("neural_network.png")
plot(nn_tem)
dev.off()


table(test_$target)

library(neuralnet)
pr.nn <- compute(nn_tem,test_[,2:45])

pr.nn_ <- pr.nn$net.result*(max(train_$target)-min(train_$target))+min(train_$target)

#test.r <- (test_$target)*(max(train_$target)-min(train_$target))+min(train_$target)

#MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

def <- ifelse(pr.nn_> 0.5,1,0)

f <- factor(def)
g <- factor(test_$target)


detach(package:neuralnet,unload = T)

confusionMatrix(data = f, reference = g)


table( test_$target, def,dnn=c("Actual","Predicted"))

misClasificError <- mean(def != test_$target)
print(paste('Accuracy',1-misClasificError))


library(ROCR)

detach(package:neuralnet,unload = T)

pr <- prediction(pr.nn_, test_$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

png("neural.png")
plot(prf)

dev.off()

