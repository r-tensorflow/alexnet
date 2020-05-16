#' @export
alexnet_install <- function() {
  reticulate::py_install("scipy")
  reticulate::py_install("pillow")
}

#' AlexNet Model
#'
#' Make use of AlexNet \link{https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
#' to train an image classifier.
#'
#' @import keras
#' @export
alexnet_train <- function(batch_size = 128,
                          epochs = 1) {

  tiny_imagenet <- pins::pin("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
  tiny_imagenet_train <- gsub("/test.*", "/train", tiny_imagenet[[1]])

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
     
    layer_dense(200) %>%
    layer_activation("softmax")

  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(momentum = 0.9, decay = 0.0005),
    metrics = "accuracy"
  )

  datagen <- image_data_generator(
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE,
    zca_whitening = TRUE
  )

  model %>% fit_generator(
    flow_images_from_directory(directory = tiny_imagenet_train,
                               batch_size = 128,
                               generator = datagen,
                               target_size = c(224, 224)),
    steps_per_epoch = as.integer(100200 / 128)
  )
}
