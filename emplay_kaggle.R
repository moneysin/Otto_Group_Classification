## install.packages('Boruta')
## library(Boruta)
library(dplyr)
library(caret)
## library(randomForest)
library(reshape2)
library(xgboost)

train_read= read.csv("D:/Tarah.AI/kaggle/train.csv/train.csv", na.strings= c('NA',""))   ## read train data
colSums(is.na(train))   ## check for missing values
classes= unique(train_read$target)   ## storing class names
 
             
test_read= read.csv("D:/Tarah.AI/kaggle/test.csv/test.csv", na.strings= c('NA',""))   ## read test data
test= subset(test_read, select=-c(id))   ## removing id from test set

train_read$target= factor(train_read$target, labels=c(0,1,2,3,4,5,6,7,8),
levels=c('Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'))  ## convert target into labels

for (i in c(2:94)){
  train_read[i]= as.numeric(unlist(train_read[i]))
}         ## convert integer columns into numeric

for (i in c(2:94)){
  test_read[i]= as.numeric (unlist(test_read[i]))
}          ## convert integer columns into numeric


train_read$target= as.numeric(train_read$target)   ## convert integer into numeric to be worked by param

train_label= train_read$target
numberOfClasses= max(train_read$target)+1   ## for passing argument into param



new_train= subset(train_read, select=-c(id,target))  ## subsetting
new_test= subset(test_read, select=-c(id))           ## subsetting

data_train= xgb.DMatrix(data=data.matrix(new_train),label= train_label)

params= list(booster="gbtree", objective="multi:softprob", eta=0.3, 
             gamma=0, num_class=numberOfClasses, eval_metric='mlogloss' , max_depth=6, min_child_weight=1, subsample=1, 
             colsample_bytree=1)      ## which has to be passed as one of the parameter in xgboost algorithm

xgbcv= xgb.cv( params = params, data = data_train, nrounds = 100, nfold = 5, showsd = T, 
               print_every_n = 10, early_stop_round = 20, maximize = F)   ## checking for number of rounds to get minimum loss

xgb1= xgboost(params = params, data = data_train, nrounds = 79, print_every_n = 10,
              early_stop_round = 20, maximize = F)   ## final algorithm

pred= predict(xgb1, data.matrix(new_test))  ## prediction

pred_mat = data.frame(matrix(pred, ncol=9, byrow=TRUE))   ## 9 columns added for each classes
colnames(pred_mat) = classes
final= data.frame(id, pred_mat)

write.csv(final, 'submission.csv', quote = F, row.names = F)

