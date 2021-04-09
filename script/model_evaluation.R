library(tidyverse)
library(tidymodels)
library(textrecipes)
library(discrim)
library(themis)
library(patchwork)

#--------------------------------------------------
# Procedure
#--------------------------------------------------

# 1. Preprocess data and specify recipe
#
# 2. Create three models
#     - A null model which guesses class with 50 % probability
#     - A naive Bayes model for binary classification
#     - A lasso regression model
#
# 3. Fit models to v-fold resamples
#
# 4. Evaluate model metrics to select best model
# 
# 5. Save chosen model for tuning


#--------------------------------------------------
# Preprocessing
#--------------------------------------------------

## Load data
english_research <- read_csv("./data/KL_research_english.csv")


## Set correct factor levels
english_research <- english_research %>% 
  mutate(kl_set = factor(if_else(kl_set == "Kjønns- og/eller likestillingsperspektiv",
                                 "Relevant",
                                 "Irrelevant")),
         kl_set = fct_rev(kl_set))


## Create training and testing set
english_split <- initial_split(english_research, strata = kl_set)
english_train <- training(english_split)
english_test <- testing(english_split)


## Create recipe
english_rec <- recipe(kl_set ~ text, data = english_train)


## Specify recipe preprocessing operations
english_rec <- english_rec %>% 
  step_downsample(kl_set) %>% 
  step_tokenize(text) %>% 
  step_tokenfilter(text, max_tokens = 1e3, min_times = 10) %>% 
  step_tfidf(text)


## Create a workflow
english_wf <- workflow() %>% 
  add_recipe(english_rec)

## Create resamples
english_folds <- vfold_cv(english_train)

#--------------------------------------------------
# Null model
#--------------------------------------------------

# Specify a simple null model which just guesses class randomly
# with a 50 % probability
null_english <- null_model() %>% 
  set_engine("parsnip") %>% 
  set_mode("classification")

set.seed(1099)
null_rs <- workflow() %>% 
  add_recipe(english_rec) %>% 
  add_model(null_english) %>% 
  fit_resamples(english_folds, 
                control = control_resamples(save_pred = TRUE))


#--------------------------------------------------
# Naive Bayes model
#--------------------------------------------------

## Specify naive Bayes model
nb_spec <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")


## Fit naive Bayes model
nb_fit <- english_wf %>% 
  add_model(nb_spec) %>% 
  fit(data = english_train)


## Resampling for model performance testing
set.seed(4567)

nb_wf <- workflow() %>% 
  add_recipe(english_rec) %>% 
  add_model(nb_spec)

nb_rs <- fit_resamples(nb_wf, 
                       english_folds, 
                       control = control_resamples(save_pred = TRUE))



#--------------------------------------------------
# Lasso model
#--------------------------------------------------

lasso_spec <- logistic_reg(penalty = 0.01, mixture = 1) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

lasso_wf <- workflow() %>% 
  add_recipe(english_rec) %>% 
  add_model(lasso_spec)

set.seed(1234)
lasso_rs <- fit_resamples(lasso_wf,
                          english_folds,
                          control = control_resamples(save_pred = TRUE))


#--------------------------------------------------
# Model evaluation
#--------------------------------------------------

windowsFonts("sans" = "Open Sans")

# Collect accuracy metrics and create plot
accuracy_plot <- collect_metrics(null_rs) %>% 
  mutate(model = "Null") %>% 
  bind_rows(collect_metrics(nb_rs) %>% 
              mutate(model = "Naive Bayes"),
            collect_metrics(lasso_rs) %>% 
              mutate(model = "Lasso")) %>% 
  filter(.metric == "accuracy") %>% 
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_pointrange(aes(ymin = mean - std_err,
                    ymax = mean + std_err),
                show.legend = FALSE) +
  theme_bw() +
  scale_color_manual(values = c("#28585a",
                                "#f7d019",
                                "#aad9dd")) +
  labs(x = NULL, y = NULL,
       title = "Mean accuracy with standard error") +
  theme(plot.title.position = "plot")

# Collect predictions and plot
roc_curve_plot <- collect_predictions(null_rs) %>% 
  mutate(model = "Null") %>%
  bind_rows(collect_predictions(nb_rs) %>% 
              mutate(model = "Naive Bayes"),
            collect_predictions(lasso_rs) %>% 
              mutate(model = "Lasso")) %>% 
  group_by(model) %>% 
  roc_curve(truth = kl_set, .pred_Relevant) %>% 
  autoplot() +
  scale_color_manual(values = c("#28585a",
                                "#f7d019",
                                "#aad9dd")) +
  labs(title = "Mean ROC curve for model performance",
       subtitle = "Larger distance from diagonal indicates better performance",
       color = NULL,
       x = "Specificity",
       y = "Sensitivity")

accuracy_plot + roc_curve_plot

ggsave("./plots/accuracy_roc_curve_plot.png", dpi = 300)


## The lasso model is chosen for further tuning, in model_tuning.R