setwd("C:/Users/33652/Desktop/ESILV/A5/ML")
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

#lecture des données
train<-read.csv("train.csv",sep = ",",nrows = 100000, stringsAsFactors = FALSE,colClasses=c("character","integer","character","character","character","character","character","character","character","integer","integer","integer")) ; 
test<-read.csv("test.csv",sep = ",",nrows = 100000,  stringsAsFactors = FALSE,colClasses=c("character","integer","character","character","character","character","character","character","character","integer","integer","integer")) ; 
# création d'une colonne indicatrice train test avant assemblage des deux tables
train$datasplit<-"train" ; test$datasplit<-"test"
# suppression d'une colonne visiblement inutile
train$campaignCode<-NULL ; test$campaignCode<-NULL
# identification des 4 colonnes au format json
json<-c("trafficSource","totals","geoNetwork","device")
tables<-c("train","test")
require(data.table)
glob<-data.table() #table vide qui va récupérer les tableas transformées
# lecture et transformation successive train et test (suppression au passage de colonnes inutiles) 
for (t in tables) 
{
  partiel<-get(t)[,setdiff(names(get(t)),json)] # colonnes non json
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


#On etudie le type de donnee de chacune des colonnes afin de convertir celles qui nous semble utile pour mieux travailler sur notre dataset
typeof(df$transactionRevenue)


summary(df)
#TRAITEMENT
#Traitement des types de donnees selon le feature
df$pageviews = as.numeric(df$pageviews)
df$page = as.numeric(df$page)
df$transactionRevenue = as.numeric(df$transactionRevenue)
df = df[,-which(names(df) %like% "datasplit")]
df$hits = as.numeric(df$hits)

#Traitement des valeurs uniques 
char<-names(df)[which(sapply(df, class)=="character")]
int<-names(df)[which(sapply(df, class)%in% c("integer","numeric"))]
level<-sort(sapply(df[,char], function(x) length(unique(x))))# identifier le nombre de valeur différentes pour les colonnes string
todelete <-level[level<2]


#On remarque a partir de la variable "Level" qu'il n'y a qu'une seule valeur associee a cette colonne. Nous ne l'utiliserons donc pas.
length(todelete)

df = df[,-match(names(todelete),names(df))]



#Traitement de la date
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


#Visualisation des NA 
isna<-sort(sapply(df, function(x) sum(is.na(x))/length(x)))
isna<-isna[isna>0]
isnaDT<-data.table(var=names(isna),txna=isna)
isnaDT[,type:="integer"] ; 
isnaDT[var %in% char,type:="string"] ; 
isnaDT[,var:=factor(var,levels=names(isna))] # pour ordonner l'affichage
isnaDT[var %in% char,type:="string"] # pour différencier la couleur en fonction du type
ggplot(isnaDT[txna>0],aes(x=var,y=txna))+geom_bar(stat="identity",aes(fill=type))+geom_text(aes(label = isnaDT$txna))+theme_tufte()

#Traitement des NA 
df$transactionRevenue[is.na(df$transactionRevenue)] <- 0
df$isVideoAd[is.na(df$isVideoAd)] <- "Unknown"
df$isTrueDirect[is.na(df$isTrueDirect)] <- "Unknown"
df$pageviews[is.na(df$pageviews)] <- 0
df$page[is.na(df$page)] <- 0


#reappply isna function
df[is.na(df)] = "Unknown"

#Etude de la valeur a predire 
typeof(df$transactionRevenue)
df_transaction = filter(df,transactionRevenue >0)
df_transaction$transactionRevenue = df_transaction$transactionRevenue/1000000
ggplot(df_transaction,aes(x=transactionRevenue)) +geom_histogram(fill="blue", binwidth=10)


#Etude de la distribution des donnees 
hist(df$date[df$date<20170000], prob = TRUE, main = "Density histogram of 2016" )
lines(density(df$date[df$date<20170000]), col = "red")


hist(df$date[20170000<df$date & df$date<20180000], prob = TRUE, main = "Density histogram of 2017")                                # Histogram and density
lines(density(df$date[20170000<df$date & df$date<20180000]), col = "red")

hist(df$date[df$date>20180000], prob = TRUE, main = "Density histogram of 2018")                                # Histogram and density
lines(density(df$date[df$date>20180000]), col = "red")

#Etude des variables numeriques 
df %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#etude des variables qualitatives 
df = df[names(df) !="date"]

char<-names(df)[which(sapply(df, class)=="character")]
df_char = df[ , which(names(df) %in% char)]
stud = ldply(df_char, function(x) length(unique(x)))
stud = stud[stud$.id != "transactionRevenue",]
study2 = stud[stud$V1<50,]
stud = stud[stud$V1>50& stud$V1<15000, ]

#plot1 <- ggplot(stud, aes(x=.id, y=V1)) + geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+geom_text(aes(label = V1),vjust=-0.3, size=3.5)+ theme_tufte()
#Second plot with features with less than 50 values
#plot2 <- ggplot(study2, aes(x=.id, y=V1)) + geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+geom_text(aes(label = V1),vjust=-0.3, size=3.5)+ theme_tufte()

#grid.arrange(plot1, plot2,nrow =2)

#Etude Network_Domain
Network_Domain = as.data.frame(table(df$networkDomain))
Network_Domain = Network_Domain[Network_Domain$Freq>1000, ]

ggplot(Network_Domain, aes(x=Var1, y=Freq)) + 
  geom_bar(stat="identity", fill="steelblue") + labs(x = "")+labs(y="")+
  geom_text(aes(label = Freq),vjust=-0.3, size=3.5)+ theme_tufte()

#Traitement Network_Domain
t = aggregate(x = df$transactionRevenue,                # Specify data column
        by = list(df$networkDomain), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 2)
set_to_other = t[which(t$x==0),]
t = t[-which(t$x<1),]
ggplot(t, aes(x=Group.1, y=x)) + 
geom_bar(stat="identity", fill="steelblue") + labs(x = "Network Domain")+labs(y="% contribution to revenue")+
geom_text(aes(label = x),vjust=-0.3, size=3.5)+ theme_tufte()
df = mutate(df, networkDomain= ifelse(networkDomain %in% set_to_other$Group.1 , "Others", networkDomain))
length(unique(df$networkDomain))

#Etude Channel_Grouping
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
  ggplot(t, aes(x=Group.1, y=x)) + 
    geom_bar(stat="identity", fill="steelblue") + labs(x = label)+labs(y="% contribution to revenue")+
    geom_text(aes(label = x),vjust=-0.3, size=3.5)+ theme_tufte()
}
Grouped_analysis(df$channelGrouping,"channelGrouping")
Grouped_analysis(df$source,"source")
Grouped_analysis(df$operatingSystem, "operatingSystem")
Grouped_analysis(df$referralPath,"referralPath")
Grouped_analysis(df$deviceCategory,"deviceCategory")
Grouped_analysis(df$browser,"browser")
Grouped_analysis(df$medium,"medium")
Grouped_analysis(df$isVideoAd,"IsVideoAd")
Grouped_analysis(df$campaign,"campaign")

#Traitement des groupes
#channel Grouping
t = aggregate(x = df$transactionRevenue,                # Specify data column
              by = list(df$channelGrouping), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$channelGrouping)))
df = mutate(df, channelGrouping= ifelse(channelGrouping %in% set_to_other$Group.1 , "Others", channelGrouping))
print(length(unique(df$channelGrouping)))

#Source
t = aggregate(x = df$transactionRevenue,                # Specify data column
              by = list(df$source), FUN = sum) 
t$x = round((t$x/sum(t$x))*100,digits = 3)
set_to_other = t[which(t$x==0),]
print(length(unique(df$source)))
df = mutate(df, source= ifelse(source %in% set_to_other$Group.1 , "Others", source))
print(length(unique(df$source)))

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

################################
#Scaling and centering data
df_gbm = df
df_gbm$transactionRevenue = log(df_gbm$transactionRevenue + 1)
df_gbm$hits = log(df_gbm$hits + 1)
df_gbm$pageviews = log(df_gbm$pageviews + 1)
df_gbm$page = log(df_gbm$page + 1)
###################################


######################
#Ridge Regression
df_ridge = df

#Scaling and centering data
df_ridge$transactionRevenue = log(df_ridge$transactionRevenue + 1)
df_ridge$hits = log(df_ridge$hits + 1)
df_ridge$pageviews = log(df_ridge$pageviews + 1)
df_ridge$page = log(df_ridge$page + 1)
num_vars = which(sapply(df_ridge, is.numeric))
to_scale = names(num_vars)
to_scale = to_scale[-which(to_scale=="transactionRevenue")]
to_scale
center_scale <- function(x) {
  scale(x, scale = FALSE)
}
mean(df_ridge$transactionRevenue[df_ridge$transactionRevenue>0])
# splitting data back to train and test 
train = df_ridge[df_ridge$transactionRevenue>0,]
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
best_lambda = ridge_model$bestTune$lambda
min(ridge_model$resample$RMSE) 
plot(ridge_model, main = "Ridge Regression") 

#Choosing the best model based on lambda
best_ridge <- glmnet(as.matrix(dplyr::select(dtrain,-contains("Id"))), trainlabel, alpha = 0, lambda = best_lambda)
best_lambda
pred_train <- predict(best_ridge, s = best_lambda, newx = as.matrix(dplyr::select(dtrain,-contains("Id"))))


dtest = mutate_if(dtest,is.character, factor)
dtest = mutate_if(dtest, is.factor, as.integer)

#Pour tester le modele sur notre test set, a enlever dans le cas ou l'on predit le vrai test set

test_transaction = dtest$transactionRevenue
dtest$transactionRevenue = NULL
dim(as.matrix(dtrain))
dim(as.matrix(dplyr::select(dtest,-contains("Id"))))
pred <- predict(best_ridge, s = best_lambda, newx = as.matrix(dplyr::select(dtest,-contains("Id"))))
output<-cbind(pred,user)
output<-mutate(output,pred = ifelse(pred < 0, 0, pred))
#output %>% group_by(fullVisitorId) %>% summarise(pred = log1p(pred))

pred_train <- ifelse(pred_train < 0, 0, pred_train)

#Comparing model's performance
rmse(trainlabel[trainlabel>0],pred_train[pred_train>0])
rmse(test_transaction[test_transaction>0],output$pred[output$pred>0])
mean(trainlabel[trainlabel>0])
#########################

#DEBUT DU MODELE FINAL CLASSIFICATION PAR ARBRE PUIS REGRESSION RIDGE 
df_classification = df_gbm
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

rf <- randomForest(as.factor(trainLabel$`dtrain$buyer`) ~ .,  
                   data = dtrain[ ,-which(names(dtrain) %in% id_toremove)],
                   ntrees = 500,
                   maxnodes = 85,
                   mtry = 6,
                   sampsize = c(600,600),
                   replace = T,
                   do.trace=100,
                   oob_score = T,
                   cutoff = c(0.55,0.45)) 

rf
importance <- rf$importance
importance
plot(rf)


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

PRROC_obj <- roc.curve(rf$votes[,1],rf$votes[,2],rand.compute = T,
                       curve=TRUE)
plot(PRROC_obj)


#Processing before Regression
output_train_1= dtrain[dtrain$output_class==0,]
names(output_train_1)[names(output_train_1)=="output_class"] <- "output_regression"
dtrain <- dtrain[dtrain$output_class==1,]
dtrain$output_class = NULL


output_test_1= dtest[dtest$output_class==0,]
names(output_test_1)[names(output_test_1)=="output_class"] <- "output_regression"
dtest <- dtest[dtest$output_class==1,]
dtest$output_class = NULL

#Regression Part 

control = trainControl(method ="cv", number = 10) 
Grid_la_reg = expand.grid(alpha = 0, 
                          lambda =seq(2, -2, by = -.1)) 
#10^seq(2, -2, by = -.1)
#seq(0.001, 0.1, by = 0.0002)
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

dtrain$output_regression <- predict(best_ridge, s = best_lambda, newx = as.matrix(dtrain[ , -which(names(dtrain) %in% id_toremove)]))
dtest$output_regression <- predict(best_ridge, s = best_lambda, newx = as.matrix(dtest[ , -which(names(dtest) %in% id_toremove)]))

dtrain$output_regression[dtrain$output_regression<0] <- 0
dtest$output_regression[dtest$output_regression<0] <- 0
#dtrain$output_regression = NULL
#dtest$output_regression = NULL

#Score of train set
rmse(dtrain$transactionRevenue,dtrain$output_regression)
output_train <- rbind(dtrain,output_train_1)
rmse(output_train$output_regression,output_train$transactionRevenue)

#Score of Test set
rmse(dtest$transactionRevenue,dtest$output_regression)
output_test <- rbind(dtest,output_test_1)
rmse(output_test$output_regression,output_test$transactionRevenue)
#For Output for Each users
#output_test %>% group_by(fullVisitorId)
