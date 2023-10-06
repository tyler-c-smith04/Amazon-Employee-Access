#################################
# Amazon Employee Access Analysis
#################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding

train <- vroom("./train.csv")
test <- vroom("./test.csv")

my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) # dummy variable encoding


# NOTE: some of these step functions are not appropriate to use together 13
# apply the recipe to your data
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = NULL)

baked
