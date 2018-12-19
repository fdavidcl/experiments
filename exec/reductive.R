#!/usr/bin/env Rscript

#library(ruta)
devtools::load_all("../../ruta")
library(purrr)
library(caret)


wdbc <- read.csv("data/wdbc.data")[, 2:32] %>% preparation("M", "M")
arcene <- read.table("data/arcene_train.data")
arcene$class <- read.table("data/arcene_train.labels")
arcene <- arcene %>% preparation()

madelon <- read.table("data/madelon_train.data")
madelon$class <- read.table("data/madelon_train.labels")
madelon <- madelon %>% preparation()

datasets <- list(
  madelon,
  wdbc
)

results <- map(datasets, function(dataset) {
  resultsnoae <- dataset %>% experiment(FALSE, "kknn")
  resultsae <- dataset %>% experiment(autoencoder, "kknn", normalize = TRUE)
  resultsaered <- dataset %>% experiment(autoencoder_reductive, "kknn", normalize = TRUE)

  list(
    baseline = resultsnoae,
    basic = resultsae,
    reductive = resultsaered
  )
})

compare <- function(index, metric) {
  map(results[[index]], function(x) mean(unlist(map(x, ~ .[[metric]]))))
}


