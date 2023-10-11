#################################
# Amazon Employee Access Analysis
#################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding

train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

test <- vroom("./test.csv") %>% 
  select(-1)

my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors()) # dummy variable encoding


# NOTE: some of these step functions are not appropriate to use together 13
# apply the recipe to your data
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = NULL)

baked

# Logistic Regression -----------------------------------------------------
library(tidymodels)

log_mod <- logistic_reg() %>% # Type of model
  set_engine('glm')

amazon_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(log_mod) %>% 
  fit(data = train) # Fit the workflow

amazon_predictions <- predict(amazon_wf,
                              new_data = test,
                              type = 'prob')

amazon_submission <- amazon_predictions %>%
  mutate(Id = row_number()) %>% 
  rename("Action" = ".pred_1") %>% 
  select(3,2)

vroom_write(x=amazon_submission, file="./logistic_reg.csv", delim=",")

# Penalized Regression

target_encoding_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # dummy variable encoding

pen_log_mod <- logistic_reg(mixture = tune() , penalty = tune() ) %>% 
  set_engine('glmnet')

pen_log_wf <- workflow() %>% 
  add_recipe(target_encoding_recipe) %>% 
  add_model(pen_log_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

## Splits data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
cv_results <- pen_log_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% 
  select_best('roc_auc')

# Finalize wf and fit it
final_wf <- 
  pen_log_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

# Predict
pen_log_preds <- final_wf %>% 
  predict(new_data = test, type = 'prob')

pen_log_submission <- pen_log_preds %>%
  mutate(Id = row_number()) %>% 
  rename("Action" = ".pred_1") %>% 
  select(3,2)

vroom_write(x=pen_log_submission, file="./pen_log_reg.csv", delim=",")
