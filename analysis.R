#!/usr/bin/env Rscript

# ---
# OVERVIEW
# AUTHOR: Brad West
# CREATED ON: 2018-11-29
# ---
# DESCRIPTION: Analysis code for ESOF 523 term paper which examines metamorphic
#              testing of random forest classifiers
# ---

library(dplyr)
library(ggplot2)
library(optparse)

option_list = list(
  make_option(c("-d", "--directory"), type="character", 
              default="~/drive/msu/fall2018/esof523/term_paper/img", 
              help="directory to store images", metavar="character"),
  make_option(c("-i", "--input"), type="character", 
              default=paste0("~/drive/msu/fall2018/esof523/term_paper/mtrf/", 
                             "data/initial_output.csv"), 
              help="input csv", metavar="character")) 

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# FIGURES =====================================================================

PROP_TRAINING_DATA <- 0.8
MRS <- 1:4
SUBTITLES <- list(
  "Linear Transform",
  "Addition of Uninformative Variable",
  "Modify Order of Predictors",
  "Increase Size of Dataset"
)
PIC_FORMAT <- paste0(opt$directory, "/scatterMr%s.png")

# read data
data <- readr::read_csv(opt$input)

d <- data %>% 
  mutate(num_test = n_samples * PROP_TRAINING_DATA,
         prop_different = n_diff / num_test,
         any_different = n_diff != 0)

for (mr in MRS) {
  num_obs <- sum(d$mr == mr)
  data_by_mr <- d[d$mr == mr, ] %>% 
    mutate(id = 1:num_obs,
           n_samples = as.factor(n_samples))
  fig <- data_by_mr %>% 
    ggplot(mapping = aes(x = id, y = prop_different, color = n_samples)) +
    geom_point() +
    theme_bw() +
    labs(title = sprintf(
      "Proportion of Differing Primary and Follow-up Test Cases, MR %s (n=72)", 
      mr),
      color = "Number of Samples",
      y = "Prop. incorrect predictions in follow-up",
      x = "Model id",
      subtitle = SUBTITLES[[mr]])
  out = sprintf(PIC_FORMAT, mr)
  ggsave(out, plot = fig, device = png(), width = 16, 
         height = 9, units = "in", dpi = 300, scale = 0.5)
  print(sprintf("Saved mr %s to %s", mr, out))
    
}

# TABLE =======================================================================

tbl <- d %>% 
  group_by(mr) %>% 
  summarise(prop_failing_tests = sum(n_diff > 0) / n()) %>% 
  rename(`MR` = mr,
         `Prop. Failing Tests` = prop_failing_tests)
xtable::xtable(tbl, caption = paste0(
  "The proportion of models fit where the predicted classes were not all", 
  " equal, out of a total of 72, for each metamorphic relation"))

# Variability in failing tests
tbl_mean_num_failing <- d %>% 
  group_by(mr) %>% 
  summarise(`Mean Prop. Failing` = mean(prop_different),
            `Std. Dev. Prop. Failing` = sd(prop_different),
            `Max Prop. Failing` = max(prop_different), 
            `Min Prop. Failing` = min(prop_different)) %>% 
  rename(MR = mr)
xtable::xtable(tbl_mean_num_failing, caption = paste0(
  "The mean and standard deviation for the number of incorrect predictions", 
  " within a given fit of the model for the follow-up test case"),
  digits = 3)

# ANALYSIS ====================================================================

# Logistic regression model using number of samples, amount of information, and
# number of classes as predictors

d2 <- d %>% 
  mutate(n_informative = as.factor(n_informative),
         n_classes = as.factor(n_classes))
model <- glm(any_different~n_samples+n_informative+n_classes, data=d2,
             family=binomial())
summary(model)

model2 <- glm(any_different~n_samples+n_classes, data=d2,
              family=binomial())
anova(model2, model, test = "Chisq")

# Moderate evidence that the odds of a failed follow-up test increase with
# increasing sample size. No evidence that amount of information or number of
# classes affect the odds of a failing follow-up test.
