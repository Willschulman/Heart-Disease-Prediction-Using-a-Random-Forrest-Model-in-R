##loading important packages
library(tidymodels)
library(tidyverse)
library(tune)
library(workflows)

##reading the data into R
dat <- read.csv('heart.csv')

dat[,"target"] <- as.factor(dat[,"target"])

##taking a look at the data
glimpse(dat)
head(dat)

##creating histograms to view variable distribution
for (i in 1:(ncol(dat)-1)) {
  print(ggplot(dat) + 
          geom_histogram(aes(dat[,i]))+
          xlab(colnames(dat)[i]))
}

##creating violin plots to see differences in distribution for the target outcomes
for (i in 1:(ncol(dat)-1)) {
  print(ggplot(dat) + 
          geom_violin(aes(dat[,i], target))+
          xlab(colnames(dat)[i]))
}

##splitting the data up into training and test sets
set.seed(123)
split_dat <- initial_split(dat)
train_dat <- training(split_dat)
test_dat <- testing(split_dat)

split_dat

##creating a cross validated version of the training set to tune parameters
dat_cv <- vfold_cv(train_dat)

rand_for_recipe <- 
  recipe(target ~ ., data = dat) %>%
  step_normalize(all_numeric())

##specifying the use of a random forest model 
rand_for_model <- rand_forest() %>%
##specifying that we will be tuning mtry
##(number of variables that can be split on at each tree node)
  set_args(mtry = tune()) %>%
##specifying the engine and the mode of variable importance
  set_engine("ranger", importance = "impurity") %>%
##selecting the mode, for ranger the options are "classification" and "regression"
  set_mode("classification")

##adding the the recipe and model to a workflow
rand_for_workflow <- 
  workflow() %>% 
  add_recipe(rand_for_recipe) %>% 
  add_model(rand_for_model)

##creating a dataframe of mtry values to test
rand_for_tune_grid <- expand_grid(mtry = c(2,3,4,5))
##calculating the tune results
rand_for_tune_results <- rand_for_workflow %>%
  tune_grid(resamples = dat_cv, 
            grid = rand_for_tune_grid, 
            ##selecting relevent metrics to assess fit
            metrics = metric_set(roc_auc, accuracy))

##displaying the results
collect_metrics(rand_for_tune_results)

##using select_best() to choose the mtry value with the best auc
rand_for_final <- 
  rand_for_tune_results %>% 
  select_best(metric = "roc_auc")

##adding the final(tuned) parameter to the workflow to finalize
rand_for_workflow <- 
  rand_for_workflow %>% 
  finalize_workflow(rand_for_final)

##using last_fit to fit the model on the training set, and evaluate it on the test set
rand_for_fit <- 
  last_fit(rand_for_workflow, split_dat)

rand_for_fit

##collecting metrics to evaluate the model
collect_metrics(rand_for_fit)

##collecting the model's prediction for each patient
model_predictions <- 
  collect_predictions(rand_for_fit)

##returning a confusion matrix of the model's predictions and actual results
model_predictions %>%
  conf_mat(target, .pred_class)

##fiting the model on entire dataset to evaluation of future data
final_model <- 
  fit(rand_for_workflow, dat)

##extracting the fit object
model_obj <- 
  pull_workflow_fit(final_model)$fit

model_obj

##examining variable importance
model_obj$variable.importance
