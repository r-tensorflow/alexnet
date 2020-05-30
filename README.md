alexnet: R Port for AlexNet Model
================

## Training

We recommend using the
[mlverse/docker](https://github.com/mlverse/docker) image which you can
retrieve and run as follows:

``` bash
docker pull mlverse/mlverse-base:version-0.2.0
docker run --gpus all -p 8787:8787 -d mlverse/mlverse-base:version-0.2.0
```

You can then connect to RStudio Server under port
<https://public-address:8787>, followed by installing and runninng
AlexNet:

``` r
remotes::install_github("mlverse/alexnet")

alexnet_train::alexnet_train()
```
