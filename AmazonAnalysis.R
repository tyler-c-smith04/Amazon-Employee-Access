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
  step_other(all_nominal_predictors(), threshold = .01) %>% 
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

