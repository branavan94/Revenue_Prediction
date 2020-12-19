## Customer Revenue Prediction


### Introduction

The stated problem is part of a customer analysis context associated with the Google Store. Indeed, in order to better target their promotions, the marketing team needs to have more information on the customers generating revenue on the Gstore (which represents according to the pareto principle 20% of the customers). To do this, we have a dataset that aims to predict the revenue generated per customer that will allow us to adjust the marketing strategy of the company.

The output file will have two columns: One to identify the visitor's ID and one that represents the predicted revenue, in log format. This file will in principle contain all the predictions of users over the period from December 1, 2018 to January 31, 2019, data that can be retrieved from the test.v2 file when we want to test our model. For training purposes, we will be able to train on earlier dates that are present on the train and test file, a smaller dataset and therefore easier to handle.

Our dataset is divided into two files: test and train
We will use the whole dataset to process and clean the data and then split our set in 60/40 to recreate our train and test set. The evaluation of the model will be done according to several selected indicators including the RMSE.

Let's study our dataset in order to better understand the constraints related with the problem.

Let's Load the Libraries first.
```{r}
set.seed(700146)
suppressPackageStartupMessages(suppressWarnings({
  library("PRROC")
  library("ggplot2")
  library("gridExtra")
  require("jsonlite") 
  library("dplyr")
  library("plyr")
  require("ggthemes")
  library('corrplot')
  library("MASS")
  library("data.table")
  library("lubridate")
  library("magrittr")
  library("ranger")
  library("stringr")
  library("lightgbm")
  library("pls")
  library("caret")
  library("purrr")
  library("tidyr")
  library("glmnet")
  library("ModelMetrics")
  library("MLmetrics")
  library("sgd")
  library("xgboost")
  library("randomForest")
}))
```
We can then read and process some data, especially the json columns that need to be processed.
```{r}
#Reading Data
train<-read.csv("train.csv",sep = ",",nrows = 100000, stringsAsFactors = FALSE,colClasses=c("character","integer","character","character","character","character","character","character","character","integer","integer","integer")) ; 
test<-read.csv("test.csv",sep = ",",nrows = 100000,  stringsAsFactors = FALSE,colClasses=c("character","integer","character","character","character","character","character","character","character","integer","integer","integer")) ; 

train$datasplit<-"train" ; test$datasplit<-"test"
# Supress Useless feature
train$campaignCode<-NULL ; test$campaignCode<-NULL
# Identify 4 columns in Json Format
json<-c("trafficSource","totals","geoNetwork","device")
tables<-c("train","test")
require(data.table)
glob<-data.table() 
# Iteratively transforming train and test
for (t in tables) 
{
  partiel<-get(t)[,setdiff(names(get(t)),json)] # Non Json Column
  for (j in json) partiel<-cbind(partiel,fromJSON(paste("[", paste(get(t)[[j]],collapse = ","), "]")))
  temp<-partiel$adwordsClickInfo
  partiel$adwordsClickInfo<-NULL
  temp$targetingCriteria<-NULL
  result<-as.data.table(cbind(partiel,temp))
  if(t=="train") result$campaignCode<-NULL else result$transactionRevenue<-NA
  glob<-rbind(glob,result)
}
rm(partiel, train, test)
gc()
df <- as.data.frame(glob)
```
Let's check the data !
```{r}
summary(df)
```
Our dataset is 2.5Gb, which is heavy and can lead to many problems when processing and analyzing this data on Rstudio. We decided to use the first 100,000 lines of our train and test dataset. Each line corresponds to one user visit, so we have 200,000 user visits to process. It should also be taken into account that a user can be represented by more than one of these lines.
For each user in the entire test, the objective is to determine "transactionRevenue", which represents the revenue per user that we are trying to predict.
This problem is indeed a regression problem as it is a question of determining a quantitative variable. Since we have enough to verify our predictions to train our model, the learning is supervised.

### Preprocessing

Let's hit the Preprocessing part.
```{r}
#Change data type according to the feature
df$pageviews = as.numeric(df$pageviews)
df$page = as.numeric(df$page)
df$transactionRevenue = as.numeric(df$transactionRevenue)
df = df[,-which(names(df) %like% "datasplit")]
df$hits = as.numeric(df$hits)
```

```{r}
#Deleting Features with only one value 
char<-names(df)[which(sapply(df, class)=="character")]
int<-names(df)[which(sapply(df, class)%in% c("integer","numeric"))]
level<-sort(sapply(df[,char], function(x) length(unique(x))))
todelete <-level[level<2]
length(todelete)
```
```{r}
df = df[,-match(names(todelete),names(df))]
```

#### Date 

Seeing the date feature, we can clearly see that we can extract some valuable informations out of it since we analyze consumers tendancy.

```{r}
dates = as.Date(as.character(df$date),"%Y%m%d")
class(dates)

x <-ymd(dates)
day = day(x)
months = month(x)
year = year(x)
indx <- setNames( rep(c('winter', 'spring', 'summer','fall'),each=3), c(12,1:11))

df$Season <- unname(indx[as.character(months)])
df$day = day
df$month = months
df$year = year
unique(df$month)
df_visitStartTime = as.POSIXct(df$visitStartTime, tz="UTC", origin='1970-01-01')
hour = hour(df_visitStartTime)
df$hour = hour
df = df[names(df) !="visitStartTime"]
```
### Process NAs

Next part : Analyze and process the NA's
```{r}
isna<-sort(sapply(df, function(x) sum(is.na(x))/length(x)))
isna<-isna[isna>0]
isnaDT<-data.table(var=names(isna),txna=isna)
isnaDT[,type:="integer"] ; 
isnaDT[var %in% char,type:="string"] ; 
isnaDT[,var:=factor(var,levels=names(isna))]
isnaDT[var %in% char,type:="string"] 
ggplot(isnaDT[txna>0],aes(x=var,y=txna))+geom_bar(stat="identity",aes(fill=type))+geom_text(aes(label = round(isnaDT$txna,digits = 3)),vjust=-0.3, size=3.5)+theme_tufte()

```
One fact is immediately obvious: less than one percent of visits lead to a transaction and therefore income for the Gstore. In the same way, many variables lack information: Gclld, page, slot, adNetworkType, isVideoAd, adContent
This can lead our model, if poorly implemented, to rely on negligible variations and potentially fall into overfitting. We must also take into account the fact that we cannot remove these characteristics either, which may very well correspond to data that modulates the very low proportion of buyers. 

Let's remove/change those NAs.
```{r}
df$transactionRevenue[is.na(df$transactionRevenue)] <- 0
df$pageviews[is.na(df$pageviews)] <- 0
df$page[is.na(df$page)] <- 0

df[is.na(df)] = "Unknown"
```
### Target Variable 

Let's look at the distribution of data among users who have potentially made transactions.
```{r}
df_transaction = filter(df,transactionRevenue >0)
df_transaction$transactionRevenue = df_transaction$transactionRevenue/1000000
ggplot(df_transaction,aes(x=transactionRevenue)) +geom_histogram(fill="blue", binwidth=10)
```
The values being very high, we had to divide this one by 10^6 to be able to obtain a scaled graph without causing vector size problems (the classical scaling function would take away too much information about the distribution of these data). The first obvious remark is that the distribution of our variable is not normal, which will make the implementation of our ML model a bit more complex. The majority of transactions are null, which confirms our previous observation about the only percentage of users making a transaction.
To implement our model, we will log-scale our data.

In order to be able to interpret these data and to have a suitable solution, it would be relevant to start by working on a random forest. Indeed, with this algorithm, it would be much easier to separate the buyers from the non buyers (internal classification) and then have a better approximation of the income per user (regression). 

### Features analysis

Before looking at the qualitative variables, we will quickly analyze the quantitative variables for possible trends associated with visitors.
Let's first look at dates ! 

```{r}
par(mfrow=c(2,2))
hist(df$date[df$date<20170000], prob = TRUE, main = "Density histogram of 2016" ,xlab="")
lines(density(df$date[df$date<20170000]), col = "red")

hist(df$date[20170000<df$date & df$date<20180000], prob = TRUE, main = "Density histogram of 2017",xlab="")
lines(density(df$date[20170000<df$date & df$date<20180000]), col = "red")

hist(df$date[df$date>20180000], prob = TRUE, main = "Density histogram of 2018",xlab="")
lines(density(df$date[df$date>20180000]), col = "red")
```
We notice that the distribution of the data depends on 3 to 4 major dates in 2016 and 2018, but a more distributed trend in 2017, which would represent more the trend of our dataset.  But let's also keep in mind that we only extracted a sample of the dataset (200,000 visits).

#### Numerical Features

We can now check at all of our numerical features.
```{r}
df %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
```
We observe several trends in visits to the Gstore according to our day/hours/month variables, particularly in the middle and end of the month, with a trend in the evening, which is quite plausible and in line with the free time of an average adult working during the day. The date being no longer useful, we will remove it soon. Visitnumber and visitId are not to be taken into account since they are only identifiers associated with a user. 

#### Categorical Features

Next, let's look at the qualitative variables and their distributions with these two graphs.
```{r}
df = df[names(df) !="date"]

char<-names(df)[which(sapply(df, class)=="character")]
df_char = df[ , which(names(df) %in% char)]
stud = ldply(df_char, function(x) length(unique(x)))
stud = stud[stud$.id != "transactionRevenue",]
study2 = stud[stud$V1<50,]
stud = stud[stud$V1>50& stud$V1<15000, ]

ggplot(stud, aes(x=.id, y=V1)) + geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+geom_text(aes(label = V1),vjust=-0.3, size=3.5)+ theme_tufte()
```
```{r}
#Second plot with features with less than 50 values
ggplot(study2, aes(x=.id, y=V1)) + geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+geom_text(aes(label = V1),vjust=-0.3, size=3.5)+ theme_tufte()

```
Our first thought was to understand the Network Domain column and the problems that can occur after the implementation of our model. 
To do so, we will simply visualize how the data is distributed by domain. For the sake of visualization, we are going to display the domains that have been displayed more than 1000 times.

```{r}
Network_Domain = as.data.frame(table(df$networkDomain))
Network_Domain = Network_Domain[Network_Domain$Freq>1000, ]

ggplot(Network_Domain, aes(x=Var1, y=Freq)) + 
  geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+
  geom_text(aes(label = Freq),vjust=-0.3, size=3.5)+ theme_tufte()

```
We identify two values that emerge from this: "Not set" and "Unknown.unknown". Despite our willingness to combine these two values to reduce the error during the classification phase, we have found no information to prove that these two values are basically the same. 

Since we have many characteristics with many different values, the idea would be to identify each of them and see their contribution to the variable explained "transactionRevenue". A modification of these characteristics (deletion or regrouping of values) will follow, especially for the qualitative variables, in order to improve the accuracy of our model.
```{r}
t = aggregate(x = df$transactionRevenue,                # Specify data column
        by = list(df$networkDomain), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 2)
set_to_other = t[which(t$x==0),]
t = t[-which(t$x<1),]
ggplot(t, aes(x=Group.1, y=x)) + 
geom_bar(stat="identity", fill="steelblue") + labs(x = "Network Domain")+labs(y="% contribution to revenue")+
geom_text(aes(label = x),vjust=-0.3, size=3.5)+ theme_tufte()
df = mutate(df, networkDomain= ifelse(networkDomain %in% set_to_other$Group.1 , "Others", networkDomain))
```
Let's calculate the % contribution of each value group of each feature based on the revenue generated.
Let's take an example with channelGrouping which contains only 6 values. 
We notice that "Referral" contributes to more than half of the total income of our dataset. On the other hand, two variables do not contribute and could be grouped in "Others". 

```{r}
Grouped_analysis = function(col_togroup, label){
  t = aggregate(x = df$transactionRevenue,                # Specify data column
                by = list(col_togroup), FUN = sum) 
  t$x = round((t$x/sum(t$x))*100,digits = 3)
  set_to_other = t[which(t$x==0),]
  print(length(set_to_other$Group.1))
  if(length(unique(t$Group.1))>12){
    other = t[which(t$x<1),]
    value = sum(other$x)
    t = t[-which(t$x<1),]
    t = rbind(t,c("Others",value))
  }
  return(ggplot(t, aes(x=Group.1, y=x)) + 
    geom_bar(stat="identity", fill="steelblue") + labs(x = label)+labs(y="% contribution to revenue")+
    geom_text(aes(label = x),vjust=-0.3, size=3.5)+ theme_tufte())
}
Grouped_analysis(df$channelGrouping,"channelGrouping")
```
```{r}
plot1 <- Grouped_analysis(df$source,"source")
plot2 <- Grouped_analysis(df$operatingSystem, "operatingSystem")

grid.arrange(plot1, plot2,nrow =2)
```
```{r}
plot3 <-Grouped_analysis(df$deviceCategory,"deviceCategory")
plot4 <-Grouped_analysis(df$referralPath,"referralPath")
grid.arrange(plot3, plot4,nrow =2)
```
```{r}
plot5 <- Grouped_analysis(df$browser,"browser")
plot6 <- Grouped_analysis(df$medium,"medium")
grid.arrange(plot5, plot6,nrow =2)
```
```{r}
plot7 <- Grouped_analysis(df$isVideoAd,"IsVideoAd")
plot8 <- Grouped_analysis(df$campaign,"campaign")
grid.arrange(plot7, plot8,nrow =2)
```
### Features Processing

Now that we see what changes we can do, let's reduce the number of unused attributes per feature.

```{r}
#channel Grouping
t = aggregate(x = df$transactionRevenue,                
              by = list(df$channelGrouping), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$channelGrouping)))
df = mutate(df, channelGrouping= ifelse(channelGrouping %in% set_to_other$Group.1 , "Others", channelGrouping))
print(length(unique(df$channelGrouping)))
```
```{r}
#Source
t = aggregate(x = df$transactionRevenue,              
              by = list(df$source), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$source)))
df = mutate(df, source= ifelse(source %in% set_to_other$Group.1 , "Others", source))
print(length(unique(df$source)))
```
```{r}

#Operating System
t = aggregate(x = df$transactionRevenue,                # Specify data column
              by = list(df$operatingSystem), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$operatingSystem)))
df = mutate(df, operatingSystem= ifelse(operatingSystem %in% set_to_other$Group.1 , "Unused", operatingSystem))
print(length(unique(df$operatingSystem)))

#Referral Path
t = aggregate(x = df$transactionRevenue,                # Specify data column
              by = list(df$referralPath), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$referralPath)))
df = mutate(df, referralPath= ifelse(referralPath %in% set_to_other$Group.1 , "Others", referralPath))
print(length(unique(df$referralPath)))

#Browser
t = aggregate(x = df$transactionRevenue,                # Specify data column
              by = list(df$browser), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$browser)))
df = mutate(df, browser= ifelse(browser %in% set_to_other$Group.1 , "Others", browser))
print(length(unique(df$browser)))
```

Based on our initial data analysis and redesign, we have come to understand that a lot of the numerical data is not centered, and that it is possible that the small percentage of this data is our best predictor of the income spent per user.
Our dataset contains a lot of different features, which can easily lead us to overfitting problems.

```{r}
#Scaling and centering data
df$transactionRevenue = log(df$transactionRevenue + 1)
df$hits = log(df$hits + 1)
df$pageviews = log(df$pageviews + 1)
df$page = log(df$page + 1)
```
### Model 1: Ridge Regression 

The first model we will use will be Ridge regression, which will be adapted for the use of many characteristics as opposed to simple linear regression, without cancelling a characteristic as would lasso regression. 
```{r}
#Ridge Regression
df_ridge = df
train = df_ridge
trainIndex <- 1:nrow(train)
data = sort(sample(nrow(train), nrow(train)*.7))
dtrain<-train[data,]
dtest<-train[-data,]

dtrain = mutate_if(dtrain,is.character, factor)
dtrain = mutate_if(dtrain, is.factor, as.integer)
user = dtest[, c("fullVisitorId", "pageviews", "bounces")]
# Model Building :Ridge regression
trainlabel = dtrain$transactionRevenue
dtrain$transactionRevenue = NULL
control = trainControl(method ="cv", number = 10) 
Grid_la_reg = expand.grid(alpha = 0, 
                          lambda =seq(0.001, 0.1, by = 0.0002)) 


ridge_model = train(x = dplyr::select(dtrain,-contains("Id")), 
                    y = trainlabel, 
                    method = "glmnet", 
                    trControl = control, 
                    tuneGrid = Grid_la_reg 
) 
```
```{r}
best_lambda = ridge_model$bestTune$lambda
min(ridge_model$resample$RMSE) 
plot(ridge_model, main = "Ridge Regression") 
```

```{r}
#Choosing the best model based on lambda
best_ridge <- glmnet(as.matrix(dplyr::select(dtrain,-contains("Id"))), trainlabel, alpha = 0, lambda = best_lambda)
best_lambda
pred_train <- predict(best_ridge, s = best_lambda, newx = as.matrix(dplyr::select(dtrain,-contains("Id"))))
pred_train <- ifelse(pred_train < 0, 0, pred_train)
```
```{r}
dtest = mutate_if(dtest,is.character, factor)
dtest = mutate_if(dtest, is.factor, as.integer)
test_transaction = dtest$transactionRevenue
dtest$transactionRevenue = NULL
pred <- predict(best_ridge, s = best_lambda, newx = as.matrix(dplyr::select(dtest,-contains("Id"))))
output<-cbind(pred,user)
output<-mutate(output,pred = ifelse(pred < 0, 0, pred))
```
```{r}
#Comparing model's performance
rmse(trainlabel[trainlabel>0],pred_train[pred_train>0])
rmse(test_transaction[test_transaction>0],output$pred[output$pred>0])
mean(trainlabel[trainlabel>0])
```
```{r}
#Model's Performance in the whole dataset
rmse(trainlabel,pred_train)
rmse(test_transaction,output$pred)
mean(trainlabel)
```
We obtain an RMSE of 1.41 for our training set and 1.43 for our test set. 
The values of our explained variable range from 0 to 22, one would think that our RMSE does not seem to be a concern.
However, we must keep in mind that our client is trying to predict the income of the buyers. Let's then look more precisely at this small portion. 
We notice immediately that if we only take the predictions >0 , we obtain an RMSE of 17.3 with an average of confirmed buyers (on the dataset of 200,000 users) of 17.8. 
Our model must then detect the buyers of simple visitors, but fails to correctly predict the amount spent by them, we're completely underfitting.

### Final Model 

Given the data unbalancing problem we were facing, coupled with an underfitting problem with our models, we therefore decided to implement two models, which would predict the class of a user as being a buyer (1 or 0), which would then serve as a second dataset for our second regression model which would only train on buyers.Note that, our priority in the classification model, is to reduce the number of False Positive (non buyer predicted as buyer) to reduce our error in the RMSE as much as possible for the regression. This reflection was done mainly by noting the very low proportion of buyers in relation to the entire dataset but also by lack of computing capacity.

```{r}
df_classification = df
df_classification$buyer = ifelse(df_classification$transactionRevenue==0,0,1)
train = df_classification

trainIndex <- 1:nrow(train)
data = sort(sample(nrow(train), nrow(train)*.7))
dtrain<-train[data,]
dtest<-train[-data,]

id_toremove = names(df)[which(str_detect(names(df[,]),"Id"))]
id_toremove <- c(id_toremove,"transactionRevenue")

user = dtest[, c("fullVisitorId", "pageviews", "bounces")]
user_train = dtrain[,c("fullVisitorId", "pageviews", "bounces")]


dtrain = mutate_if(dtrain,is.character, factor)
categorical_feature <- names(Filter(is.factor, dtrain[,-which(names(dtrain) %in% id_toremove)]))
dtrain = mutate_if(dtrain, is.factor, as.integer)


data = sort(sample(nrow(dtrain),nrow(dtrain)*0.4))
dvalid = dtrain[data,]

trainLabel <- as.data.frame(dtrain$buyer)
trainLabel$regression <- dtrain$transactionRevenue
valLabel = dvalid$buyer
dtrain$buyer = NULL
dvalid$buyer = NULL

```

The Random forests parameters had been optimized through a 10-fold Cross-validation. 
Below, we can find importance of each features for our classifcation.
```{r}
rf <- randomForest(as.factor(trainLabel$`dtrain$buyer`) ~ .,  
                   data = dtrain[ ,-which(names(dtrain) %in% id_toremove)],
                   ntrees = 500,
                   maxnodes = 85,
                   mtry = 6,
                   sampsize = c(600,600),
                   replace = T,
                   do.trace=100,
                   oob_score = T,
                   cutoff = c(0.50,0.50)) 

rf
importance <- rf$importance
importance
```
```{r}
plot(rf)
```
```{r}
PRROC_obj <- roc.curve(rf$votes[,1],rf$votes[,2],rand.compute = T,
                       curve=TRUE)
plot(PRROC_obj)
```

```{r}
dtrain$output_class  <- predict(rf, as.matrix(dtrain[ , -which(names(dtrain) %in% id_toremove)], type= "class"))
dtest = mutate_if(dtest,is.character, factor)
dtest = mutate_if(dtest, is.factor, as.integer)
testlabel = data.frame(dtest$transactionRevenue)
testlabel$buyer = dtest$buyer
testlabel$fullVisitorId = dtest$fullVisitorId

dtest$buyer = NULL

dtest$output_class<- predict(rf, as.matrix(dtest[ , -which(names(dtest) %in% id_toremove)], type= "class"))

auc(as.numeric(dtrain$output_class),trainLabel$`dtrain$buyer`)
auc(as.numeric(dtest$output_class),as.numeric(testlabel$buyer))
```
Finally, we arrive by performing a grid search of some parameters with a 4.3% error, a result that remains correct but that will still cause problems during the regression.
```{r}
#Processing before Regression
output_train_1= dtrain[dtrain$output_class==0,]
names(output_train_1)[names(output_train_1)=="output_class"] <- "output_regression"
dtrain <- dtrain[dtrain$output_class==1,]
dtrain$output_class = NULL


output_test_1= dtest[dtest$output_class==0,]
names(output_test_1)[names(output_test_1)=="output_class"] <- "output_regression"
dtest <- dtest[dtest$output_class==1,]
dtest$output_class = NULL
```
```{r}
#Regression Part 

control = trainControl(method ="cv", number = 10) 
Grid_la_reg = expand.grid(alpha = 0, 
                          lambda =seq(2, -2, by = -.1)) 
ridge_model = train(x = dtrain[ , -which(names(dtrain) %in% id_toremove)], 
                    y = dtrain$transactionRevenue, 
                    method = "glmnet",
                    metric = "RMSE",
                    trControl = control, 
                    tuneGrid = Grid_la_reg 
) 
best_lambda = ridge_model$bestTune$lambda
plot(ridge_model, main = "Ridge Regression") 
best_ridge <- glmnet(as.matrix(dtrain[ , -which(names(dtrain) %in% id_toremove)]), dtrain$transactionRevenue, alpha = 0, lambda = best_lambda)

```
```{r}
dtrain$output_regression <- predict(best_ridge, s = best_lambda, newx = as.matrix(dtrain[ , -which(names(dtrain) %in% id_toremove)]))
dtest$output_regression <- predict(best_ridge, s = best_lambda, newx = as.matrix(dtest[ , -which(names(dtest) %in% id_toremove)]))

dtrain$output_regression[dtrain$output_regression<0] <- 0
dtest$output_regression[dtest$output_regression<0] <- 0

#Score of train set
rmse(dtrain$transactionRevenue,dtrain$output_regression)
output_train <- rbind(dtrain,output_train_1)
rmse(output_train$output_regression,output_train$transactionRevenue)

#Score of Test set
rmse(dtest$transactionRevenue,dtest$output_regression)
output_test <- rbind(dtest,output_test_1)
rmse(output_test$output_regression,output_test$transactionRevenue)

#To Output for Each users
#output_test %>% group_by(fullVisitorId)
```

For our regression part, we use the ridge regression with our optimized lambda parameter. 
We obtain for our test data 5.7 of RMSE (1.6 for the total test dataset, including the non buyer), a better result than our previous model but which remains problematic for a regression.
 
### Perspectives
 
In order to improve our model, we propose some lines of research. First, our data has a big data unbalancing problem, a problem that could be solved if we collected enough data on buyers and had more balanced data. This could be done through the use of modified subset, which would greatly reduce our dataset size, but would be feasible given the size of the entire dataset (~30GB).
Most of the remaining work lies in our accuracy in classifying a buyer and a non-buyer. By minimizing our FPR rate, and maximizing our TP and TN, we would have a better view on the fit of our regression model. Other means could be put in place to artificially increase our data, notably through new samples calculated from existing values. 

