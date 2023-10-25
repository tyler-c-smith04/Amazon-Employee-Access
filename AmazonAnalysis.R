#################################
# Amazon Employee Access Analysis
#################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding
library(ranger)
library(rpart)
library(discrim)
library(naivebayes)
library(kknn)
library(doParallel)

parallel::detectCores()
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

test <- vroom("./test.csv") %>% 
  select(-1)

# my_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   step_dummy(all_nominal_predictors()) # dummy variable encoding

predict_and_format <- function(workflow, new_data, filename){
  predictions <- workflow %>%
    predict(new_data = new_data,
            type = "prob")
  
  submission <- predictions %>%
    mutate(Id = row_number()) %>% 
    rename("Action" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(x = submission, file = filename, delim=",")
}


# # NOTE: some of these step functions are not appropriate to use together 13
# # apply the recipe to your data
# prepped_recipe <- prep(my_recipe)
# baked <- bake(prepped_recipe, new_data = NULL)
# 
# baked

# Logistic Regression -----------------------------------------------------
# library(tidymodels)
# # 
# log_mod <- logistic_reg() %>% # Type of model
#   set_engine('glm')
# 
# amazon_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(log_mod) %>%
#   fit(data = train) # Fit the workflow
# 
# amazon_predictions <- predict(amazon_wf,
#                               new_data = test,
#                               type = 'prob')
# 
# amazon_submission <- amazon_predictions %>%
#   mutate(Id = row_number()) %>%
#   rename("Action" = ".pred_1") %>%
#   select(3,2)
# 
# vroom_write(x=amazon_submission, file="./logistic_reg.csv", delim=",")

# Penalized Regression

target_encoding_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% # dummy variable encoding
  step_normalize(all_nominal_predictors())
# 
# 
# pen_log_mod <- logistic_reg(mixture = tune() , penalty = tune() ) %>%
#   set_engine('glmnet')
# 
# pen_log_wf <- workflow() %>%
#   add_recipe(target_encoding_recipe) %>%
#   add_model(pen_log_mod)
# 
# ## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5)
# 
# ## Splits data for CV
# folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# cv_results <- pen_log_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# best_tune <- cv_results %>%
#   select_best('roc_auc')
# 
# # Finalize wf and fit it
# final_wf <-
#   pen_log_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data = train)
# 
# # Predict
# pen_log_preds <- final_wf %>%
#   predict(new_data = test, type = 'prob')
# 
# pen_log_submission <- pen_log_preds %>%
#   mutate(Id = row_number()) %>%
#   rename("Action" = ".pred_1") %>%
#   select(3,2)
# 
# vroom_write(x=pen_log_submission, file="./pen_log_reg.csv", delim=",")


# # Random Forests ----------------------------------------------------------
# rand_forest_mod <- rand_forest(mtry = tune(),
#                                min_n=tune(),
#                                trees=500) %>% # or 1000
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# rand_forest_workflow <- workflow() %>%
#   add_recipe(target_encoding_recipe) %>%
#   add_model(rand_forest_mod)
# 
# rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
#                                         min_n(),
#                                         levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# forest_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- rand_forest_workflow %>%
#   tune_grid(resamples = forest_folds,
#             grid = rand_forest_tuning_grid,
#             metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy
# 
# ## Find Best Tuning Parameters
# forest_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_forest_wf <- rand_forest_workflow %>%
#   finalize_workflow(forest_bestTune) %>%
#   fit(data = train)
# 
# predict_and_format(final_forest_wf, test, "./random_forest_predictions.csv")

# Naive Bayes -------------------------------------------------------------
# library(discrim)
# 
# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode('classification') %>%
#   set_engine('naivebayes')
# 
# nb_wf <- workflow() %>%
#   add_recipe(target_encoding_recipe) %>%
#   add_model(nb_model)
# 
# # Tune smoothness and Laplace here
# nb_tuning_grid <- grid_regular(Laplace(),
#                                smoothness(),
#                                levels = 5)
# 
# ## Split data for CV
# nb_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- nb_wf %>%
#   tune_grid(resamples = nb_folds,
#             grid = nb_tuning_grid,
#             metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy
# 
# ## Find Best Tuning Parameters
# nb_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_nb_wf <- nb_wf %>%
#   finalize_workflow(nb_bestTune) %>%
#   fit(data = train)
# 
# # Predict
# predict(final_nb_wf, new_data = test, type = 'prob')
# 
# predict_and_format(final_nb_wf, test, "./nb_preds.csv")
# 
# # k-nearest neighbors -----------------------------------------------------
# 
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# knn_wf <- workflow() %>%
#   add_recipe(target_encoding_recipe) %>%
#   add_model(knn_model)
# 
# # cross validation
# knn_tuning_grid <- grid_regular(neighbors(),
#                                 levels = 5)
# 
# knn_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- knn_wf %>%
#   tune_grid(resamples = knn_folds,
#             grid = knn_tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# knn_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # finalize workflow
# final_knn_wf <- knn_wf %>%
#   finalize_workflow(knn_bestTune) %>%
#   fit(data = train)
# 
# predict_and_format(final_knn_wf, test, "./knn_predictions.csv")

# Principal Component Dimension Reduction ---------------------------------
# pcdr_recipe <- recipe(ACTION ~ ., train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .001) %>%  # combines categorical values that occur <1% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # target encoding (must be 2-factor)
#   step_normalize(all_nominal_predictors()) %>% 
#   step_pca(all_predictors(), threshold = .8) #Threshold is between 0 and 1

# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode('classification') %>%
#   set_engine('naivebayes')
# 
# nb_wf <- workflow() %>%
#   add_recipe(pcdr_recipe) %>%
#   add_model(nb_model)
# 
# # Tune smoothness and Laplace here
# nb_tuning_grid <- grid_regular(Laplace(),
#                                smoothness(),
#                                levels = 5)
# 
# ## Split data for CV
# nb_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- nb_wf %>%
#   tune_grid(resamples = nb_folds,
#             grid = nb_tuning_grid,
#             metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy
# 
# ## Find Best Tuning Parameters
# nb_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_nb_wf <- nb_wf %>%
#   finalize_workflow(nb_bestTune) %>%
#   fit(data = train)
# 
# # Predict
# predict(final_nb_wf, new_data = test, type = 'prob')
# 
# predict_and_format(final_nb_wf, test, "./nb_pcdr_preds.csv")
# 
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# knn_wf <- workflow() %>%
#   add_recipe(pcdr_recipe) %>%
#   add_model(knn_model)
# 
# # cross validation
# knn_tuning_grid <- grid_regular(neighbors(),
#                                 levels = 5)
# 
# knn_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- knn_wf %>%
#   tune_grid(resamples = knn_folds,
#             grid = knn_tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# knn_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # finalize workflow
# final_knn_wf <- knn_wf %>%
#   finalize_workflow(knn_bestTune) %>%
#   fit(data = train)
# 
# predict_and_format(final_knn_wf, test, "./knn_pcdr_predictions.csv")

# Random Forests w PCDR ---------------------------------------------------
# rand_forest_mod <- rand_forest(mtry = tune(),
#                                min_n=tune(),
#                                trees=1000) %>% # or 1000
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# rand_forest_workflow <- workflow() %>%
#   add_recipe(pcdr_recipe) %>%
#   add_model(rand_forest_mod)
# 
# rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
#                                         min_n(),
#                                         levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# forest_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- rand_forest_workflow %>%
#   tune_grid(resamples = forest_folds,
#             grid = rand_forest_tuning_grid,
#             metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy
# 
# ## Find Best Tuning Parameters
# forest_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_forest_wf <- rand_forest_workflow %>%
#   finalize_workflow(forest_bestTune) %>%
#   fit(data = train)
# 
# predict_and_format(final_forest_wf, test, "./random_forest_pcdr_predictions.csv")

# Support Vector Machines -------------------------------------------------
svmPoly <- svm_poly(degree = tune(), cost = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('kernlab')

svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('kernlab')

svmLinear <- svm_linear(cost = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('kernlab')

svm_wf <- workflow() %>%
  add_recipe(target_encoding_recipe) %>%
  add_model(svmRadial)

# cross validation
svm_tuning_grid <- grid_regular(cost(),
                                rbf_sigma(),
                                levels = 5)

svm_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

svm_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_svm_wf <- knn_wf %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data = train)

predict_and_format(final_svm_wf, test, "./svm_predictions.csv")

stopCluster(cl)



