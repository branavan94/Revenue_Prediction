#DEBUT Test du Modele LGBM
train = df_gbm
#[df_gbm$transactionRevenue>0,]
trainIndex <- 1:nrow(train)
data = sort(sample(nrow(train), nrow(train)*.7))
dtrain<-train[data,]
dtest<-train[-data,]

#Traitement des id
id_toremove = names(df)[which(str_detect(names(df[,]),"Id"))]
dtrain = dtrain[ , -which(names(dtrain) %in% id_toremove)]

dtrain = mutate_if(dtrain,is.character, factor)
categorical_feature <- names(Filter(is.factor, dtrain))
dtrain = mutate_if(dtrain, is.factor, as.integer)
user = dtest[, c("fullVisitorId", "pageviews", "bounces")]

data = sort(sample(nrow(dtrain),nrow(dtrain)*0.4))
dvalid = dtrain[data,]

trainLabel <- dtrain$transactionRevenue
valLabel = dvalid$transactionRevenue
dtrain$transactionRevenue = NULL
dvalid$transactionRevenue = NULL
lgb.train = lgb.Dataset(data=as.matrix(dtrain),label=trainLabel, categorical_feature =categorical_feature,silent = T)
lgb.valid = lgb.Dataset(data=as.matrix(dvalid),label=valLabel, categorical_feature =categorical_feature,silent = T)
params <- list(objective="regression",
               metric="rmse",
               learning_rate=0.01)
lgb.normalizedgini = function(preds, dtrain){
  actual = getinfo(dtrain, "label")
  score  = NormalizedGini(preds,actual)
  return(list(name = "gini", value = score, higher_better = TRUE))
}
lgb.model.cv = lgb.cv(params = params, data = lgb.train, learning_rate = 0.02, num_leaves = 25,
                      num_threads = 2 , nrounds = 7000, early_stopping_rounds = 50,
                      eval_freq = 20, eval = lgb.normalizedgini,
                      categorical_feature = categorical_feature, nfold = 10, stratified = TRUE)
best_iter <- lgb.model.cv$best_iter
#best.iter = lgb.model.cv$best_iter

lgb.model <- lgb.train(params = params,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate=0.02,
                       nrounds=best_iter,
                       verbose=1,
                       early_stopping_rounds=50,
                       feature_fraction = 0.7,
                       eval_freq=100)

tree_imp <- lgb.importance(lgb.model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 50, measure = "Gain")


pred_train <- predict(lgb.model, as.matrix(dtrain))

dtest = mutate_if(dtest,is.character, factor)
dtest = mutate_if(dtest, is.factor, as.integer)
pred <- predict(lgb.model, as.matrix(dplyr::select(dtest,-contains("Id"))))


output = cbind(pred,user)
output = mutate(output,pred = ifelse(pred < 0, 0, pred))
pred_train <-ifelse(pred_train < 0, 0, pred_train)

#pred_train= ifelse(pred_train<0.5,0,1)
#output = mutate(output,pred = ifelse(pred <0.5,0,1))

#output %>% group_by(fullVisitorId) %>% summarise(pred = log1p(pred))
rmse(trainLabel[trainLabel>0],pred_train[pred_train>0])
rmse(dtest$transactionRevenuepred_train,output$pred)
dtest$transactionRevenue

#Fin Test du Modele
########################################

#ARBRE A DECISION
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
lgb.train = lgb.Dataset(data=as.matrix(dtrain[ ,-which(names(dtrain) %in% id_toremove)]),label=trainLabel$`dtrain$buyer`, categorical_feature =categorical_feature)
lgb.valid = lgb.Dataset(data=as.matrix(dvalid[,-which(names(dvalid) %in% id_toremove)]),label=valLabel, categorical_feature =categorical_feature)
params <- list(objective="binary",
               metric="auc",
               learning_rate=0.02,
               feature_fraction = 0.7,
               booster= "rf")

lgb.model <- lgb.train(params = params,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate=0.05,
                       nrounds=500,
                       verbose=1,
                       early_stopping_rounds=50,
                       eval_freq=100)

#tree_imp <- lgb.importance(lgb.model, percentage = TRUE)
#lgb.plot.importance(tree_imp, top_n = 50, measure = "Gain")
dtrain$output_class <- predict(lgb.model, as.matrix(dtrain[ , -which(names(dtrain) %in% id_toremove)]))

dtest = mutate_if(dtest,is.character, factor)
dtest = mutate_if(dtest, is.factor, as.integer)
testlabel = data.frame(dtest$transactionRevenue)
testlabel$buyer = dtest$buyer
testlabel$fullVisitorId = dtest$fullVisitorId

dtest$buyer = NULL
dtest$output_class <- predict(lgb.model, as.matrix(dtest[ , -which(names(dtest) %in% id_toremove)]))

dtrain <- mutate(dtrain,output_class = ifelse(output_class < 0.5, 0, 1))
dtest <- mutate(dtest,output_class = ifelse(output_class < 0.5, 0, 1))

auc(dtrain$output_class,trainLabel$`dtrain$buyer`)
auc(dtest$output_class,testlabel$buyer)
################m###
#Fin du Test du Modele





