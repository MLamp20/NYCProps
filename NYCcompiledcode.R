#######################################################################
#Compiled R Code NYC Property Sales - Harvard edX Data Science Capstone
#######################################################################

###########################
# Download File from Kaggle
###########################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(corrr)) install.packages("corrr", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(readr)
library(lubridate)
library(corrr)
library(broom)
library(matrixStats)

#New York City Property Sales Sept 2016 Sept 2017 Machine Learning Repository
#https://www.kaggle.com/new-york-city/nyc-property-sales


#Necessary to download Kaggle to GitHub, then download GitHub file to RStudio
download.file("https://raw.githubusercontent.com/MLamp20/NYCProps/main/nyc-rolling-sales.csv",destfile="./Projects/data/raw/nyc_rolling_sales.csv")

nyc_rolling_sales<-read_csv("./Projects/data/raw/nyc_rolling_sales.csv")
nyc<-nyc_rolling_sales

###################################################
# Preprocessing and Exploratory Data Analysis (EDA)
###################################################

#Convert to data frame, review structure, class types, and dimensions
nyc<-as.data.frame(nyc)
str(nyc)
table(sapply(nyc,class))
dim(nyc)

#rename variables
nyc<-nyc%>% rename(BLDG_CL_CATEGORY="BUILDING CLASS CATEGORY",TAX_CL_0917="TAX CLASS AT PRESENT",
                   BLDG_CL_0917="BUILDING CLASS AT PRESENT",APT_NO="APARTMENT NUMBER",ZIP="ZIP CODE",
                   RESID_UNITS="RESIDENTIAL UNITS",COMM_UNITS="COMMERCIAL UNITS",TTL_UNITS="TOTAL UNITS",
                   LAND_SF="LAND SQUARE FEET",GRSS_SF="GROSS SQUARE FEET",BUILT="YEAR BUILT",
                   TAX_CL_SOLD="TAX CLASS AT TIME OF SALE",BLDG_CL_SOLD="BUILDING CLASS AT TIME OF SALE",
                   SALE_PX="SALE PRICE",SALE_DATE="SALE DATE",EASEMENT="EASE-MENT")
names(nyc)

#count distinct values
nyc_dist<-summarize_all(nyc,n_distinct,na.rm=TRUE)
sort(nyc_dist,decreasing = TRUE)

#count zeroes
reszero<-round(colSums(nyc==0)/nrow(nyc) *100)
sort(reszero[!reszero==0],decreasing=TRUE)

#count na by column
nyc_na<-round(colSums(is.na(nyc))/nrow(nyc) *100)
sort(nyc_na[(nyc_na>0)], decreasing = TRUE)

#convert to numerics for square footage (SF) and sale price
i<-c(16,17,21)
nyc[,i]<-apply(nyc[,i],2,function(x) as.numeric(as.character(x)))

#convert sale date to date time for set up of week of sale
nyc<-nyc%>%mutate(SL_DATE=mdy_hm(SALE_DATE),SL_WEEK=round_date(SL_DATE,unit="week")) %>%
  select(-c(SALE_DATE,SL_DATE))

#remove sale price that are blank or zero
nyc<-nyc%>%filter(!(SALE_PX<1))

#determine what percentage of bldg cl category has missing gross square footage 
nyc_na_grsf<-nyc%>%group_by(BLDG_CL_CATEGORY)%>%summarise(na_pct=round(sum(is.na(GRSS_SF))/n(),2)) %>% 
  mutate(tier=ifelse(na_pct>.75,1,ifelse(na_pct>.50,2,ifelse(na_pct>.25,3,ifelse(na_pct>0,4,5)))))
nyc_na_grsf%>%arrange(desc(na_pct))%>%print(n=nrow(.))

#review bldg cl category mean and median sales in Thousands(K) and gross sf
bcc_naz<-nyc %>% group_by(BLDG_CL_CATEGORY)%>%summarize(n=n(),nagsf=sum(is.na(GRSS_SF)),
                                                        zgsf=sum(GRSS_SF<1),accgsf=n-nagsf-zgsf)
print(bcc_naz,n=Inf)

#calculate avg grss sf by bldg cl category
bcc_ntna<-nyc %>% filter(!is.na(GRSS_SF))%>%filter(!GRSS_SF<1)%>% group_by(BLDG_CL_CATEGORY)%>%
  summarize(n=n(),avg_SL_K=round(mean(SALE_PX)/1000),med_SL_K=round(median(SALE_PX)/1000),avg_GRSF=round(mean(GRSS_SF)),med_GRSF=round(median(GRSS_SF)),avg_pxGSF=round((avg_SL_K*1000)/avg_GRSF),med_pxGSF=round((med_SL_K*1000)/med_GRSF))
print(bcc_ntna,n=Inf)

#replace BUILT with AGE, add SUM UNIT remove easement x1 
nyc<-nyc%>%mutate(AGE=2017-BUILT,SUM_UNIT=RESID_UNITS + COMM_UNITS)%>%select(-c(X1,EASEMENT,BUILT))

#replace NA in tax class and building class 0917
nyc$TAX_CL_0917[(is.na(nyc$TAX_CL_0917))]<-as.character(nyc$TAX_CL_SOLD[is.na(nyc$TAX_CL_0917)])
nyc$BLDG_CL_0917[(is.na(nyc$BLDG_CL_0917))]<-nyc$BLDG_CL_SOLD[is.na(nyc$BLDG_CL_0917)]

#replace the NA blank and zero values for land sf, gross sf.  Replace age of 2017( original data was zero)
#determine mean values for LAND_SF, GRSS_SF, AGE for records with values not NA blank zero
baseline_landsf<-nyc%>%filter(!is.na(SALE_PX))%>%filter(!is.na(LAND_SF))%>%filter(!SALE_PX<1) %>%filter(!LAND_SF<1) %>% summarize(lf=sum(SALE_PX)/sum(LAND_SF))
baseline_landsf

baseline_grsf<-nyc%>%filter(!is.na(SALE_PX))%>%filter(!is.na(GRSS_SF))%>%filter(!SALE_PX<1) %>%filter(!GRSS_SF<1) %>% summarize(lf=sum(SALE_PX)/sum(GRSS_SF))
baseline_grsf

baseline_age<-nyc%>%filter(!AGE==2017)%>%summarize(age=mean(AGE))
baseline_age

#replace NA with calculated means above
nyc<-nyc %>% dplyr::mutate(LAND_SF = replace_na(LAND_SF, baseline_landsf$lf))
nyc<-nyc %>% dplyr::mutate(GRSS_SF = replace_na(GRSS_SF, baseline_grsf$lf))
dim(nyc)

#replace zeroes with calculated means above
#replace SUM UNIT
indz<-which(nyc$SUM_UNIT<1)
nyc$SUM_UNIT[indz]<-1

#replace LAND_SF
ind<-which(nyc$LAND_SF<1)
nyc$LAND_SF[ind]<-baseline_landsf$lf

#replace GRSS_SF
index<-which(nyc$GRSS_SF<1)
nyc$GRSS_SF[index]<-baseline_grsf$lf

#replace age 2017
inda<-which(nyc$AGE==2017)
nyc$AGE[inda]<-baseline_age$age

#remove duplicated lines 
indexd<-which(duplicated(nyc))
nyc<-nyc[-indexd,]
dim(nyc)

##Visualization

#Distribution of Sale Price LOG 10
slpxdistr<-nyc%>%ggplot(aes(SALE_PX))+geom_histogram(bins=30,color="black")+ scale_x_log10()+ggtitle("Sale Price Distribution")
slpxdistr

#Data has long tail, do log conversion to remove skew

#log transform SALE PX 
nyc<-nyc%>%mutate(SALE_LOG=log(nyc$SALE_PX))

#standarize SALE LOG
nyc<-nyc%>%mutate_at("SALE_LOG",~ scale(.) %>% as.vector)

#plot transformed and standardized Sale Price
slpxdistrrev<-nyc%>%ggplot(aes(SALE_LOG))+geom_histogram(bins=30,color="black")+ggtitle("STD Sale Price Distribution")
slpxdistrrev

#prepare for further filtering of dataset

#review list of outliers
evalout<-nyc%>%filter(abs(SALE_LOG)>4)%>%select(c(SALE_PX,SALE_LOG))

#filter out values with absolute value greater than 4
nyc<-nyc%>%filter(!abs(SALE_LOG)>4)
dim(nyc)

#review revised filtered plot
nyc_wo_outlier<-nyc%>%ggplot(aes(SALE_LOG))+geom_histogram(bins=30,color="black")+ggtitle("STD Sale Price Distribution")
nyc_wo_outlier

#additional visualization
#gross sf vs SALE PX
nyc%>%ggplot(aes(GRSS_SF,SALE_PX))+geom_point(alpha=0.3,color="blue")

#boxplot visualizations vs SALE LOG
#building class category plot ordered by mean price
nyc%>%mutate(BLDG_CL_CATEGORY=reorder(BLDG_CL_CATEGORY,SALE_LOG,FUN=mean)) %>%ggplot(aes(BLDG_CL_CATEGORY,SALE_LOG))+geom_boxplot()+theme(axis.text.x = element_text(angle=90,hjust=1))

#plot SUM UNIT vs SALE LOG
boxplot(SALE_LOG~SUM_UNIT,data=nyc,horizontal=TRUE)

#plot BOROUGH vs SALE LOG
boxplot(SALE_LOG~BOROUGH,data=nyc,horizontal=TRUE)

#review sales price by borough by time
px_time<-nyc%>%group_by(SL_WEEK,BOROUGH)%>%summarize(avg_px_wk=mean(SALE_PX))%>%ggplot(aes(SL_WEEK,avg_px_wk)) +
  geom_point()+theme(axis.text.x = element_text(angle=90,hjust=1))+geom_smooth()+facet_wrap(~BOROUGH)+theme(axis.text.x = element_text(angle=90,hjust=1))
px_time

#remove SL WEEK variable
nyc<-nyc%>%select(-c(SL_WEEK))

#Correlation review
cor(nyc$SALE_PX,nyc$BOROUGH)
cor(nyc$SALE_PX,nyc$BLOCK)
cor(nyc$SALE_PX,nyc$LOT)
cor(nyc$SALE_PX,nyc$RESID_UNITS)
cor(nyc$SALE_PX,nyc$COMM_UNITS)
cor(nyc$SALE_PX,nyc$SUM_UNIT)
cor(nyc$SALE_PX,nyc$LAND_SF)
cor(nyc$SALE_PX,nyc$GRSS_SF)
cor(nyc$SALE_PX,nyc$AGE)
cor(nyc$SALE_PX,nyc$TAX_CL_SOLD)

#zip correlation where var not zero
nyc_zip<-nyc%>%filter(ZIP>0)
dim(nyc_zip)
cor(nyc_zip$SALE_PX,nyc_zip$ZIP)

#correlation borough block zip to evaluate overlap for potential variable trim in dataset model
cor(nyc$BOROUGH,nyc$BLOCK)
cor(nyc$BOROUGH,nyc$ZIP)

#correlation at sf level 
cor(nyc$GRSS_SF,nyc$RESID_UNITS)
cor(nyc$GRSS_SF,nyc$SUM_UNIT)
cor(nyc$GRSS_SF,nyc$LAND_SF)

cor(nyc$LAND_SF,nyc$RESID_UNITS)
cor(nyc$LAND_SF,nyc$SUM_UNIT)

#remove num columns with low correlation to sale px plus TTL_UNITS replaced with SUM UNIT, and APT_NO due to 77% NA
nyc<-nyc%>%select(-c(APT_NO,RESID_UNITS,COMM_UNITS,TTL_UNITS,AGE,ZIP,LOT))

#remove chr columns too granular(address) or post sale metrics (0917)
nyc<-nyc%>%select(-c(ADDRESS,TAX_CL_0917,BLDG_CL_0917))
dim(nyc)

#review neighborhood category avg price and avg price per gross square foot grouped by borough
dset_b<-nyc %>% group_by(BOROUGH)%>%summarize(n=n(),nbrhd=n_distinct(NEIGHBORHOOD),categ=n_distinct(BLDG_CL_CATEGORY),
                                              blk=n_distinct(BLOCK),avg_px_K=round(mean(SALE_PX)/1000),avg_pxgsf=round(avg_px_K*1000/mean(GRSS_SF)))
dset_b

#review neighborhood category avg price and avg price per gross square foot grouped by bldg cl category
dset_c<-nyc %>% group_by(BLDG_CL_CATEGORY)%>%summarize(n=n(),boro=n_distinct(BOROUGH),nbrhd=n_distinct(NEIGHBORHOOD),
                                                       blk=n_distinct(BLOCK),avg_px_K=round((mean(SALE_PX)/1000)),
                                                       avg_pxgsf=round(avg_px_K*1000/mean(GRSS_SF)))

dset_c_trim<- nyc %>% group_by(BLDG_CL_CATEGORY)%>%summarize(n=n(),avg_px_K=round((mean(SALE_PX)/1000)),
                                                             avg_pxgsf=round(avg_px_K*1000/mean(GRSS_SF)))

#review largest n transactions
most_active_cat<-dset_c%>%arrange(desc(n))%>%head(10)
most_active_cat
sum(most_active_cat$n)/nrow(nyc)

#review highest avg price per gross foot against info on what percentage of GRSS SF was input as mean due to NA
slsf_dollars<-nyc %>% group_by(BLDG_CL_CATEGORY)%>% summarize(TTL_SLS_K=round(sum(SALE_PX)/1000),avg_pxgsf=round(mean(SALE_PX)/mean(GRSS_SF)))
tiernagsf<-nyc_na_grsf%>%left_join(slsf_dollars,by="BLDG_CL_CATEGORY")
tiernagsf%>%mutate(BLDG_CL_CATEGORY=reorder(BLDG_CL_CATEGORY,avg_pxgsf), FUN=avg_pxgsf) %>%ggplot(aes(avg_pxgsf,BLDG_CL_CATEGORY))+ geom_point(aes(col=tier)) 


#review categories with highest sale per gsf
most_slsf_cat<-dset_c%>%arrange(desc(avg_pxgsf))%>%head(10)
cat<-dset_c_trim %>% arrange(desc(avg_pxgsf))%>%head(10)
cat %>% left_join(nyc_na_grsf,by="BLDG_CL_CATEGORY")%>%select(-c(avg_px_K))

#recap categories by top dollar sales
slsf_dollars_top<-slsf_dollars%>%arrange(desc(TTL_SLS_K))%>%head(10)
slsf_dollars_top
sum(slsf_dollars_top$TTL_SLS_K*1000)/sum(nyc$SALE_PX)

#review categoris by lowest dollar sales
slsf_dollars_bottom<-slsf_dollars%>%arrange(TTL_SLS_K)%>%head(10)
slsf_dollars_bottom

#remove block as variable - no value in hot encode 11K columns, no value in log transform standardizing
nyc<-nyc%>%select(-BLOCK)
#remove SALE PX as variable - redundant due to introduction of SALE LOG vector
nyc<-nyc %>%select(-SALE_PX)

#log transform and standardize num variables with a viable mean (vs. ordinal)
nyc$LAND_SF<-log(nyc$LAND_SF)
nyc$GRSS_SF<-log(nyc$GRSS_SF)
nyc$SUM_UNIT<-log(nyc$SUM_UNIT)

nyc<-nyc%>%mutate_at(c("LAND_SF","GRSS_SF","SUM_UNIT"),~ scale(.) %>% as.vector)
nyc%>%head()

#review gross sf scatterplot for identification and filter out of outliers
nyc%>%ggplot(aes(GRSS_SF,SALE_LOG))+geom_point(alpha=0.3,color="blue")
nyc<-nyc%>%filter(GRSS_SF<7.5)
nyc%>%ggplot(aes(GRSS_SF,SALE_LOG))+geom_point(alpha=0.1,color="blue")
dim(nyc)

#convert boro and tx cl sold to factor for use in one hot
nyc$BOROUGH<-as.factor(nyc$BOROUGH)
nyc$TAX_CL_SOLD<-as.factor(nyc$TAX_CL_SOLD)
str(nyc)

#one-hot to convert all chr vectors to columns to evaluate corr to sales price
dummy <- dummyVars(" ~ .", data=nyc)
newdata <- data.frame(predict(dummy, newdata = nyc)) 
dim(newdata)

#set up review of all correlation to SALE LOG
tbl_df<-newdata%>%correlate() %>% focus(SALE_LOG)
options(digits = 3)

#review short list of variables with greater than 10 percent correlation
corsub_for_hot<-tbl_df%>%filter(abs(SALE_LOG)>.10)
print(corsub_for_hot,n=Inf)

#add SALE LOG to tibble in prep to filter the dataset for final pre-split
corsub_for_hot<-corsub_for_hot%>%add_row(tibble_row(term="SALE_LOG","SALE_LOG"=1.0))

#extract BLDG_CL_SOLDR4 and BLDG_CL_SOLDRS due to NA with GLM model
corsub_for_hot_adj<-corsub_for_hot[c(1:22,24,26:27),]

#review final variable list
corsub_for_hot_adj$term

#reset nyc for final model
nyc_pre_split<-newdata%>%select_if(names(.)%in% corsub_for_hot_adj$term)
dim(nyc_pre_split)


############################################
# Partition Data Training vs Testing Subsets
############################################

#Test set will be 20% of data as set up for random forest model later
set.seed(1,sample.kind = "Rounding")
test_index<-createDataPartition(y=nyc_pre_split$SALE_LOG,times=1, p=0.2, list=FALSE)
train<-nyc_pre_split[-test_index,]
test<-nyc_pre_split[test_index,]

################
# Model Fitting
################


#linear regression model

#train model on training set
model<-lm(SALE_LOG ~ ., data = train)

#show intercept and value estimates per variables on train set
print(tidy(model),n=Inf)

#variable importance linear model
as.data.frame(varImp(model))%>%arrange(desc(Overall))

#review RMSE on train
rmse_train_lm<-summary(model)$sigma
rmse_train_lm

#predict values on test set
y_hat_lm<-predict.lm(model,newdata = test)

#evaluate rmse predictions for test set vs actuals
rmse_lm<-RMSE(y_hat_lm,test$SALE_LOG)

#place RMSE results on test set in data frame
rmse_result <- data_frame(Method = "Linear Regression Predictor", RMSEtrain = rmse_train_lm, RMSEtest = rmse_lm )
rmse_result%>%knitr::kable()


##rpart model

#set seed for replication of results 
set.seed(1991,sample.kind="Rounding")

#train rpart model on training set, consider tuning parameter complexity parameter (cp)
train_rpart <- train(SALE_LOG ~ ., method = "rpart", tuneGrid=data.frame(cp=seq(0,0.05,0.002)),data = train)

#review best cp value
train_rpart$bestTune

#due to length of names in dataset, plot and text unreadable, review final model decision tree in text
train_rpart$finalModel

#variable importance rpart model
varImp(train_rpart)

#review RMSE on training set
rmse_train_rpart<-train_rpart$results$RMSE[which.min(train_rpart$results$RMSE)]
rmse_train_rpart

#predict values on test set
y_hat_rpart <- predict(train_rpart, test, type = "raw")

#evaluate rmse predictions for test set vs actuals
rmse_rpart<-RMSE(y_hat_rpart,test$SALE_LOG)

#add RMSE results on test set for this model to existing data frame
rmse_result <- bind_rows(rmse_result,
                         data_frame(Method="rpart cp = .002",RMSEtrain=rmse_train_rpart,
                                    RMSEtest = rmse_rpart))
rmse_result %>% knitr::kable()

##random Forest model

#set seed for replication of results 
set.seed(14,sample.kind = "Rounding")

#train rf model on training set, ntree = 64 for processing time, consider tuning parameter (mtry)
fit_rf <- train(SALE_LOG ~ ., method = "rf",                       
                tuneGrid = data.frame(mtry = 9:12),ntree=64, data = train)

#review bestTune
fit_rf$bestTune

#review model
fit_rf

#review the importance of the top 20 variables in the buildout of the training model
varImp(fit_rf)

#review RMSE on training set
rmse_train_rf<-fit_rf$results$RMSE[which.min(fit_rf$results$RMSE)]
rmse_train_rf

#predict values on test set
y_hat_rf<-predict(fit_rf, test, type = "raw")

#evaluate rmse predictions for test set vs actuals
rmse_rf<-RMSE(y_hat_rf,test$SALE_LOG)

#add RMSE results on test set for this model to existing data frame
rmse_result <- bind_rows(rmse_result,
                         data_frame(Method="randomForest ntree=64, mtry=11",  
                                    RMSEtrain=rmse_train_rf, RMSEtest = rmse_rf))
rmse_result %>% knitr::kable()


