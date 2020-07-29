alexnet_model <- function(output) {
  model <- keras_model_sequential()

  model %>%
    layer_conv_2d(filters = 96, kernel_size = c(11, 11),
                  strides = c(4, 4), input_shape = c(224, 224, 3), padding = "valid") %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%

    layer_conv_2d(filters = 256, kernel_size = c(5, 5), strides = c(1, 1), padding = "valid") %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%

    layer_conv_2d(filters = 384, kernel_size = c(3, 3), strides = c(1, 1), padding = "valid") %>%
    layer_activation("relu") %>%

    layer_conv_2d(filters = 384, kernel_size = c(3, 3), strides = c(1, 1), padding = "valid") %>%
    layer_activation("relu") %>%

    layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(1, 1), padding = "valid") %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%

    layer_flatten() %>%
    layer_dense(4096) %>%
    layer_activation("relu") %>%
    layer_dropout(rate = 0.5) %>%

    layer_dense(4096) %>%
    layer_activation("relu") %>%
    layer_dropout(rate = 0.5) %>%

    layer_dense(output) %>%
    layer_activation("softmax")
}

alexnet_compile <- function(model) {
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(momentum = 0.9, decay = 0.0005),
    metrics = "accuracy"
  )
}

alexnet_tinyimagenet <- function() {
  tiny_imagenet <- pins::pin("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
  tiny_imagenet_train <- gsub("/test.*", "/train", tiny_imagenet[[1]])

  tiny_imagenet_subset <- dir(tiny_imagenet_train, recursive = TRUE, pattern = "*.JPEG", full.names = TRUE)
  tiny_imagenet_categories <- gsub("^.*/train/|/images.*$", "", tiny_imagenet_subset)

  category <- as.character(tiny_imagenet_categories)

  list(
    image = tiny_imagenet_subset,
    category = category,
    categories = unique(category)
  )
}

#' AlexNet Model
#'
#' Make use of AlexNet \link{https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
#' to train an image classifier.
#'
#' @import keras
#' @import tfdatasets
#' @export
alexnet_train <- function(batch_size = 32L,
                          epochs = 2,
                          data = NULL,
                          strategy = NULL) {
  if (is.null(data)) {
    data <- alexnet_tinyimagenet()
  }

  data_map <- 0:(length(data$categories)-1)
  names(data_map) <- data$categories
  data_y <- unname(sapply(data$category, function(e) data_map[e]))

  tiny_imagenet_data <- tibble::tibble(
    img = data$image,
    cat = to_categorical(data_y, length(data$categories))
  )

  if (identical(strategy, NULL)) {
    model <- alexnet_model(output = length(data_map))
    alexnet_compile(model)
  }
  else {
    with (strategy$scope(), {
      model <- alexnet_model(output = length(data_map))
      alexnet_compile(model)
    })
  }

  random_bsh <- function(img) {
    img %>%
      tf$image$random_brightness(max_delta = 0.3) %>%
      tf$image$random_contrast(lower = 0.5, upper = 0.7) %>%
      tf$image$random_saturation(lower = 0.5, upper = 0.7) %>%
      # make sure we still are between 0 and 1
      tf$clip_by_value(0, 1)
  }

  create_dataset <- function(data, train, batch_size = 32L) {
    dataset <- data %>%
      tensor_slices_dataset() %>%
      dataset_map(~.x %>% purrr::list_modify(
        img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = 3)
      )) %>%
      dataset_map(~.x %>% purrr::list_modify(
        img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)
      )) %>%
      dataset_map(~.x %>% purrr::list_modify(
        img = tf$image$resize(.x$img, size = shape(224, 224))
      ))

    if (train) {
      dataset <- dataset_map(dataset, ~.x %>% purrr::list_modify(
        img = random_bsh(.x$img)
      ))
    }

    if (train) {
      dataset_shuffle(dataset, buffer_size = batch_size*128)
    }

    dataset_batch(dataset, batch_size) %>%
      dataset_map(unname) # Keras needs an unnamed output.
  }

  model %>% fit(
    create_dataset(tiny_imagenet_data, TRUE, batch_size = batch_size),
    epochs = epochs)
}
