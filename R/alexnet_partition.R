#' Retrieve ImageNet Partition
#'
#' Retrieves a partition to be used with \code{imagenet_train} as the \code{data}
#' parameter.
#'
#' @partition The partition to retrieve.
#' @total The total number of partitions.
#' @url The url of the board containing imagenet.
#'
#' @export
imagenet_partition <- function(partition, total = 16, url = "https://storage.googleapis.com/r-imagenet/") {
  pins::board_register(url, "imagenet")
  categories <- pins::pin_get("categories", board = "imagenet")

  partition_size <- ceiling(length(categories$id) / total)
  categories <- categories$id[(partition_size * (partition - 1)) + (1:partition_size)]

  procs <- lapply(categories, function(cat)
    callr::r_bg(function(cat) {
      library(pins)
      board_register("https://storage.googleapis.com/r-imagenet/", "imagenet")

      pin_get(cat, board = "imagenet", extract = TRUE)
    }, args = list(cat))
  )

  while (any(sapply(procs, function(p) p$is_alive()))) Sys.sleep(1)

  list(
    image = unlist(lapply(categories, function(cat) {
      pins::pin_get(cat, board = "imagenet", download = FALSE)
    })),
    category = unlist(lapply(categories, function(cat) {
      rep(cat, length(pins::pin_get(cat, board = "imagenet", download = FALSE)))
    })),
    categories = categories
  )
}
