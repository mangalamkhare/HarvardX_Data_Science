# R SCript - CYO Project - Ramen Rating

#data can be downloaded from kaggle https://www.kaggle.com/residentmario/ramen-ratings
#     OR 
#GitHub "https://raw.githubusercontent.com/mangalamkhare/HarvardX_Data_Science/main/ramen-ratings.csv"

# Dataset Preparation

#############################################################
# Create Ramen data set
#############################################################

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")

if(!require(FNN)) install.packages("FNN")
if(!require(mltools)) install.packages("mltools")

if(!require(plyr)) install.packages("plyr")
if(!require(ggpubr)) install.packages("ggpubr")


library(tidyverse)
library(caret)
library(data.table)
library(FNN)
library(ggplot2)
library(lubridate)
library(dslabs)
library(plyr)
library(ggpubr)
library(class)
library(rpart)
library(randomForest)

# Ramen Ratings dataset:

link<-'https://raw.githubusercontent.com/mangalamkhare/HarvardX_Data_Science/main/ramen-ratings.csv'

df <- read.csv(file =link)

head(df)

df_ramen <- as.data.frame(df) %>% mutate(
reviewId = 'Review #' ,
topten = as.character("Top Ten"), 
brand = as.character(Brand), 
variety = as.character(Variety), 
style = as.character(Style), 
country = as.character(Country),
stars= as.numeric(Stars))
head(df_ramen)

# Data pre-processing and exploratory analysis

head(df_ramen)

#Check Dimensions and Summary stats

# Rows Columns

dim(df_ramen)

# Data set Summary

summary(df_ramen)

# check for the number of unique brand, style, variety and country in the ramen dataset

# Unique number of Style and Country 

df_ramen %>%
summarize(n_style = n_distinct(style), 
n_country = n_distinct(country))

# unique number of brand and variety  

df_ramen %>%
summarize(n_brand = n_distinct(brand), 
n_variety = n_distinct(variety))

# Since reviewId represents unique review it will not be used for modelling, we will drop it. 

# check topten column

n_topten <- unique(df_ramen$topten)

n_topten

# since topten does not have any useful data we will drop this as well

df_ramen <- df_ramen %>%  select(brand , variety, style,country, stars) 

head(df_ramen)

# Clean the data set

df_ramen [df_ramen == "Unrated"] <- "0"
df_ramen <- df_ramen  %>%na.omit()

## Define the function for RMSE

RMSE <- function(true_ratings, predicted_ratings){
sqrt(mean((true_ratings-predicted_ratings)^2))
}

## Stars/Ratings distribution

# distribution of stars

ggplot(df_ramen, aes(x=stars)) + 
geom_histogram(aes(y=..density..),     
  binwidth=0.2,
  colour="black", fill="grey")

# Brands distribution  
df_ramen %>% group_by(brand) %>% filter(n() > 30) %>% 
ggplot(aes(x = brand) )+ geom_bar() + coord_flip()


# style distribution
df_ramen %>%
ggplot(aes(style)) +
geom_bar() +
ylab("Count")

df_ramen<- df_ramen %>% 
filter(style %in% c("Pack", "Bowl", "Cup", "Tray"))

# Country Distribution

df_ramen %>% group_by(country) %>% filter(n() > 50) %>% 
ggplot(aes(x = country)) + geom_bar() + coord_flip()

y <- count(df_ramen, 'country') 
df_country <-  y[order(-y$freq),]
df_country

df_country <- df_country %>% 
filter(freq > 50)

df_country

df_ramen$country[!df_ramen$country  %in% c("Japan","USA","South   Korea",
             "Taiwan","Thailand", "China",
             "Malaysia", "Hong Kong","Indonesia",
              "Singapore","Vietnam")] <- "Others"

unique(df_ramen$country)

## Data Preparation

# First take backup of the entire data set

ramen <- df_ramen

df_ramen <- df_ramen %>% select(style, country, stars)

# Create dummy variables 

dummy <- dummyVars(" ~ .", data = df_ramen)

df_ramen_dummy <- data.frame(predict(dummy, newdata = df_ramen))

# split the data set into training and test sets
# train set- 80%, test set/validation set - 20%

set.seed(1, sample.kind="Rounding") 

test_index <- createDataPartition(y = df_ramen_dummy$stars, times = 1, p = 0.2, list = FALSE)

df_ramen_train<- df_ramen_dummy[-test_index,]
df_ramen_test <- df_ramen_dummy[test_index,]

# Modelling and analysis

# Base : Average Stars/Rating model

# Compute the mean rating from the ramen train data set

mu <- mean(df_ramen_train$stars)
mu

# Test Results based on base prediction

base_rmse <- RMSE(df_ramen_test$stars, mu)
base_rmse

# Check results save prediction in dataframe

rmse_results <- data_frame(Method = " Average Stars/Rating model", 
            RMSE = base_rmse)
rmse_results %>% knitr::kable()

# Liner Regression

set.seed(1, sample.kind = "Rounding")

train_lm <- lm(stars ~ ., data = df_ramen_train)

summary(train_lm)

prediction <- predict(train_lm,df_ramen_test)

#prediction

rmse <- RMSE(prediction, df_ramen_test$stars)

rmse
rmse_results <- bind_rows(rmse_results,
           data_frame(Method="Linear Regression model",  
                      RMSE = rmse))
rmse_results %>% knitr::kable()


# Decision Tree
set.seed(1, sample.kind = "Rounding")
train_rpart <- rpart(stars~., method = "anova", data = df_ramen_train)

train_rpart

prediction <- predict(train_rpart, df_ramen_test)

rmse <- RMSE(prediction, df_ramen_test$stars)


rmse

rmse_results <- bind_rows(rmse_results,
           data_frame(Method="Decision Tree Regression model",  
                      RMSE = rmse))
rmse_results %>% knitr::kable()


# Random Forest

set.seed(1, sample.kind = "Rounding")
train_rf <- randomForest(stars~., data = df_ramen_train, mtry = 2,
#train_rf <- randomForest(stars~., data = df_ramen_train, mtry = seq(1:7),
          importance = TRUE )

train_rf
 
prediction <- predict(train_rf, data = df_ramen_test)

rmse <- RMSE(prediction, df_ramen_test$stars)

rmse

rmse_results <- bind_rows(rmse_results,
           data_frame(Method="Random Forest Regression model",  
                      RMSE = rmse))
rmse_results %>% knitr::kable()

# Knn Regression

set.seed(1, sample.kind = "Rounding")
train_knn <- train(stars ~ .,  method = "knn", 
    tuneGrid = data.frame(k = seq(3, 8, 0.25)),
    #tuneGrid = data.frame(k = seq(3, 5, 0.25)), 
    data = df_ramen_train)


prediction <- predict(train_knn,df_ramen_test)

#prediction

rmse<-RMSE(prediction, df_ramen_test$stars)

rmse

rmse_results <- bind_rows(rmse_results,
           data_frame(Method="Knn Regression model",  
                      RMSE = rmse))
rmse_results %>% knitr::kable()



ggplot(train_knn)

train_knn$bestTune

#since we are not able to Improve RMSE's further lets try to convert     
#regression into classification, predicting if Ramen noodles are good or not
#we will consider stars>3.75 as 1 (Good) and < as 0 (Not Good)

# Data Preparation for Classification models

df_ramen_train_cl <- df_ramen_train

df_ramen_test_cl <- df_ramen_test

head(df_ramen_train_cl)
head(df_ramen_test_cl)

df_ramen_train_cl <- mutate(df_ramen_train_cl , isGood = ifelse(df_ramen_train_cl$stars > 3.75, 1, 0))
df_ramen_test_cl <- mutate(df_ramen_test_cl , isGood = ifelse(df_ramen_test_cl$stars > 3.75, 1, 0))

# we will drop stars column

df_ramen_train_cl <- df_ramen_train_cl %>%  select(-stars) 
df_ramen_test_cl <- df_ramen_test_cl %>%  select(-stars)

head(df_ramen_train_cl)
head(df_ramen_test_cl)


# Classification Models    

# LDA 

set.seed(1, sample.kind = "Rounding")
train_lm <- lm(isGood ~ ., data = df_ramen_train_cl)

summary(train_lm)

prediction <- predict(train_lm,df_ramen_test_cl)

# prediction

rmse <- RMSE(prediction, df_ramen_test_cl$isGood)

rmse
rmse_results <- bind_rows(rmse_results,
              data_frame(Method="LDA Classification model",  
                         RMSE = rmse))
rmse_results %>% knitr::kable()

# Knn Classification
set.seed(1, sample.kind = "Rounding") 
train_knn_cl <- train(isGood ~ .,
       method = "knn",
       data = df_ramen_train_cl,
       #tuneGrid = data.frame(k = seq(1, 7, 0.25)))
       tuneGrid = data.frame(k = seq(3, 8, 0.25)))
       #tuneGrid = data.frame(k = seq(3, 5, 0.25)))

prediction <- predict(train_knn_cl, df_ramen_test_cl)

rmse <- RMSE(prediction, df_ramen_test_cl$isGood)

rmse
rmse_results <- bind_rows(rmse_results,
              data_frame(Method="Knn Classification model",  
                         RMSE = rmse))
rmse_results %>% knitr::kable()

train_knn_cl$bestTune

ggplot(train_knn_cl)

# Cross Validation
set.seed(1, sample.kind = "Rounding")  
train_knn_cv_cl <- train(isGood ~ .,
          method = "knn",
          data = df_ramen_train_cl,
          #tuneGrid = data.frame(k = seq(1, 7, 0.25)),
          tuneGrid = data.frame(k = seq(3, 8, 0.25)),
          trControl = trainControl(method = "cv", number = 10, p = 0.9))

prediction <- predict(train_knn_cv_cl, df_ramen_test_cl)

rmse <- RMSE(prediction, df_ramen_test_cl$isGood)

rmse
rmse_results <- bind_rows(rmse_results,
              data_frame(Method="Cross validation Classification model",  
                         RMSE = rmse))
rmse_results %>% knitr::kable()

train_knn_cv_cl$bestTune

# Classification Tree
set.seed(1, sample.kind = "Rounding") 
train_rpart_cl <- train(isGood ~ ., 
         method = "rpart",
         tuneGrid = data.frame(cp = seq(0, 0.5, 0.02)),
         data = df_ramen_train_cl)

prediction <- predict(train_rpart_cl, df_ramen_test_cl)

rmse <- RMSE(prediction, df_ramen_test_cl$isGood)

rmse
rmse_results <- bind_rows(rmse_results,
              data_frame(Method="Classification tree model",  
                         RMSE = rmse))
train_rpart_cl$bestTune
rmse_results %>% knitr::kable()


