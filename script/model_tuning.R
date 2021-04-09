library(tidyverse)
library(tidymodels)
library(textrecipes)
library(discrim)
library(themis)
library(patchwork)
library(vip)
windowsFonts("sans" = "Open Sans")

#--------------------------------------------------
# Procedure
#--------------------------------------------------


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
  step_tokenfilter(text, 
                   max_tokens = tune(), 
                   min_times = 10) %>% 
  step_tfidf(text)

# specify model with tunable parameter
tune_spec <- logistic_reg(penalty = tune(),
                          mixture = 1) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

## Create a workflow
english_wf <- workflow() %>% 
  add_recipe(english_rec)

## Create resamples
english_folds <- vfold_cv(english_train)


# create grid for hyperparameter grid search
final_grid <- grid_regular(penalty(range = c(-4, 9)),
                           max_tokens(range = c(1e3, 3e3)),
                           levels = c(penalty = 20, max_tokens = 3))

tune_wf <- workflow() %>% 
  add_recipe(english_rec) %>% 
  add_model(tune_spec)

set.seed(9876)
tune_rs <- tune_grid(tune_wf,
                     english_folds,
                     grid = final_grid,
                     metrics = metric_set(accuracy, sensitivity, specificity),
                     control = control_resamples(save_pred = TRUE))


#--------------------------------------------------
# Evaluate performance
#--------------------------------------------------

# Select simplest model that doesn't result in unacceptable loss
chosen_auc <- tune_rs %>% 
  select_by_pct_loss(metric = "accuracy", -penalty)


final_lasso <- finalize_workflow(tune_wf, chosen_auc)

# Fit model on full test set
final_fitted <- last_fit(final_lasso, english_split)

# Plot final ROC curve and confusion matrix
final_roc_curve <- collect_predictions(final_fitted) %>% 
  roc_curve(truth = kl_set, .pred_Relevant) %>% 
  autoplot() +
  labs(x = "Specificity",
       y = "Sensitivity",
       title = "ROC curve for gender research classifier",
       subtitle = "With final tuned lasso regularized classifier on full test set") +
  theme(plot.subtitle = element_text(size = rel(0.8)))

final_confusion_matrix <- collect_predictions(final_fitted) %>% 
  conf_mat(truth = kl_set, estimate = .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "#28585a") +
  theme_bw() +
  theme(legend.position = "none") +
  labs(title = "Confusion matrix on the test set")

final_roc_curve + final_confusion_matrix

ggsave("./plots/roc_curve_confusion_matrix_plot.png", dpi = 300)

# Determine 
pull_workflow_fit(final_fitted$.workflow[[1]]) %>% 
  vi(lambda = chosen_auc$penalty) %>% 
  mutate(Sign = case_when(Sign == "POS" ~ "Less about gender research topics",
                          TRUE ~ "More about gender research topics"),
         Importance = abs(Importance),
         Variable = str_remove_all(Variable, "tfidf_text_")) %>% 
  group_by(Sign) %>% 
  top_n(20, Importance) %>% 
  ungroup() %>% 
  ggplot(aes(x = Importance,
             y = fct_reorder(Variable, Importance),
             fill = Sign)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Sign, scales = "free") +
  scale_x_continuous(expand = c(0,0)) +
  scale_fill_manual(values = c("#28585a", "#f7d019")) +
  labs(y = NULL,
       title = "Variable importance for predicting the topic of a research article",
       subtitle = paste("Most important features for determining whether", 
                        "research features gender research topics")) +
  theme_minimal() +
  theme(plot.title = element_text(margin = margin(5, 0, 5, 0)),
        plot.title.position = "plot",
        plot.subtitle = element_text(margin = margin(0, 0, 10, 0)),
        panel.grid.major.y = element_blank(),
        strip.text = element_text(hjust = 0))

ggsave("./plots/vip_plot.png", dpi = 300)


## Fit final model on all data
final_lasso_fit <- fit(final_lasso, data = english_research)

## Save model for prediction purposes
saveRDS(final_lasso_fit, "./data/KL_classifier_model.rds")
