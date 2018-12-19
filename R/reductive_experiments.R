
name <- function(features) {
  colnames(features) <- paste0("h", 1:dim(features)[2])
  features
}

train_model <- function(autoencoder_f, method, train_x, train_y, normalized) {

  if (is.logical(autoencoder_f) && autoencoder_f == FALSE) {
    feature_extractor <- function(x) return(x)
    features <- train_x
  } else {
    ## Extract features
    hidden_dim <- ceiling(0.1 * dim(train_x)[2])
    activation <- if (normalized) "sigmoid" else "linear"
    network <- input() + dense(hidden_dim, "relu") + output(activation)

    # Do not use binary crossentropy (and sigmoid activation) *unless* the data has been
    # accordingly normalized (to the [0, 1] interval)
    loss <- if (normalized) "binary_crossentropy" else "mean_squared_error"
    feature_extractor <- autoencoder_f(network, loss = loss)
    print(feature_extractor)
    feature_extractor <- if (is_reductive(feature_extractor))
      feature_extractor %>% ruta::train(train_x, classes = as.numeric(train_y) - 1, epochs = 200)
    else
      feature_extractor %>% ruta::train(train_x, epochs = 200)

    feature_extractor <- purrr::compose(name, purrr::partial(ruta::encode, learner = feature_extractor, .lazy = FALSE))
    features <- feature_extractor(train_x)
  }

  str(features)

  ## Classifier
  # ctrl <- trainControl()
  classifier <- caret::train(features, train_y, method = method)
  print(classifier)

  list(
    feature_extractor = feature_extractor,
    classifier = classifier
  )
}

test_model <- function(model, test_x) {
  features <- model$feature_extractor(test_x)
  str(features)
  predictions <- model$classifier %>% predict(newdata = features)
}

evaluate_model <- function(true_y, pred_y) {
  tp <- sum(true_y == pred_y & true_y == 1)
  tn <- sum(true_y == pred_y & true_y == 0)
  fp <- sum(true_y != pred_y & true_y == 0)
  fn <- sum(true_y != pred_y & true_y == 1)

  list(
    accuracy = mean(true_y == pred_y),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp),
    precision = tp / (tp + fp),
    fscore = 2 * tp / (2 * tp + fp + fn)
  )
}

experiment <- function(dataset, autoencoder_f, method, normalize = TRUE) {
  set.seed(4242)

  ## Prepare dataset
  k <- 5
  train_idx <- createFolds(dataset$y, k = k)
  message(str(train_idx))

  results <- list()

  for (i in 1:k) {
    train_x <- dataset$x[-train_idx[[i]],]
    train_y <- dataset$y[-train_idx[[i]]]
    test_x <- dataset$x[train_idx[[i]],]
    test_y <- dataset$y[train_idx[[i]]]

    if (normalize) {
      mx <- apply(train_x, 2, max)
      mn <- apply(train_x, 2, min)
      range <- mx - mn
      train_x <- t(apply(train_x, 1, function(x) (x - mn) / range))
      test_x <- t(apply(test_x, 1, function(x) (x - mn) / range))
    }

    model <- train_model(autoencoder_f, method, train_x, train_y, normalized = normalize)
    predictions <- test_model(model, test_x)
    results[[i]] <- evaluate_model(test_y, predictions)
  }

  results
}

preparation <- function(dataset, class_name = "class", value_positive = 1) {
  list(
    x = as.matrix(dataset[, -which(names(dataset) == class_name)]),
    y = as.factor(ifelse(dataset[, class_name] == value_positive, 1, 0))
  )
}
