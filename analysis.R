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
              default="~/drive/msu/fall2018/esof523/term_paper/mtrf/data/initial_output.csv", 
              help="input csv", metavar="character")) 

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

PROP_TRAINING_DATA <- 0.8
MRS <- 1:4
SUBTITLES <- list(
  "Linear Transform",
  "Addition of Uninformative Variable",
  "Modify Order of Predictors",
  "Increase Size of Dataset"
)
PIC_FORMAT <- paste0(opt$directory, "/scatter_mr_%s.png")

# read data
data <- readr::read_csv(opt$input)

d <- data %>% 
  mutate(num_test = n_samples * PROP_TRAINING_DATA,
         prop_different = n_diff / num_test)

for (mr in MRS) {
  num_obs <- sum(d$mr == mr)
  data_by_mr <- d[d$mr == mr, ] %>% 
    mutate(id = 1:num_obs,
           n_samples = as.factor(n_samples))
  fig <- data_by_mr %>% 
    ggplot(mapping = aes(x = id, y = prop_different, color = n_samples)) +
    geom_point() +
    theme_bw() +
    labs(title = sprintf("Proportion of Differing Primary and Follow-up Test Cases, MR %s (n=72)", mr),
         subtitle = SUBTITLES[[mr]])
  out = sprintf(PIC_FORMAT, mr)
  ggsave(out, plot = fig, device = png(), width = 16, 
         height = 9, units = "in", dpi = 800, scale = 0.5)
  print(sprintf("Saved mr %s to %s", mr, out))
    
}
