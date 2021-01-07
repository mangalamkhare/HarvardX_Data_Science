# Mangalam Khare
# MovieLens Project 
# HarvardX PH125.9x
# Dataset

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
# title = as.character(title),
# genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


#movielens <- left_join(ratings, movies, by = "movieId")

ratings_tab <- as.data.frame(ratings) %>% mutate(movieId = as.numeric(movieId), 
                                                 userId = as.numeric(userId), 
                                                 rating = as.numeric(rating), 
                                                 timestamp = as.numeric(timestamp))

movielens <- left_join(ratings_tab, movies, by = "movieId")

head(movielens)

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Additional Libraries
library(ggplot2)
library(lubridate)
library(dslabs)

# Check few rows of data set
head(edx) 


# Check Dimensions ans Summary stats

dim(edx)
summary(edx)

# check for the number of unique movies and users in the edx dataset 
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

#  Define the function for RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2,na.rm=TRUE))
}


# Ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 200000))) +
  ggtitle("Ratings distribution")

#User's Distribution

edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 25, color = "blue") + 
  scale_x_log10() + 
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

# Movies distribution

edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

#Simple : Average movie rating model

# Compute the mean rating from the edx data set

mu <- mean(edx$rating)
mu

# Test results based on simple prediction
simple_rmse <- RMSE(validation$rating, mu)
simple_rmse


# Check results
# Save prediction in data frame
rmse_results <- data_frame(method = " model using mean only", 
                           RMSE = simple_rmse)
rmse_results %>% knitr::kable()


## Movie effect model ##

# Simple model taking into account the movie effect b_i
# Plot number of movies with the computed b_i

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 15, 
                     data = ., color = I("blue"),
                     ylab = "Number of movies",
                     main = "Number of movies with the computed b_i")



# Test and save rmse results 
predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse))
# Check results
rmse_results %>% knitr::kable()

rmse_results

# 3rd Model considering Movie efect and user effect 

# Plot penaly term user effect #
predicted_ratings_user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings_user_avgs %>% qplot(b_u, geom ="histogram", bins = 25, data = ., color = I("blue"))


predicted_ratings_user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Test and save rmse results 
predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(predicted_ratings_user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model",  
                                     RMSE = model_2_rmse))

# Check result
rmse_results %>% knitr::kable()

# lambda is a regularization parameter
# We need to start by taking a few lambda values starting from zero
lambdas <- seq(0, 10, 0.25)


# For each lambda,find b_i & b_u, followed by rating prediction & testing

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


# Plot rmses vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  


# The optimal lambda                                                             
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                     RMSE = min(rmses)))

# Check result
rmse_results %>% knitr::kable()

#### Results ####                                                            
# RMSE results overview                                                          
rmse_results %>% knitr::kable()


