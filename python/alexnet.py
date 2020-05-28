import time

from tensorflow import keras
from tensorflow.keras import layers

tiny_imagenet_train = "/home/rstudio/.cache/pins/local/tiny_imagenet_200/tiny-imagenet-200/train"

model = keras.Sequential()

model.add(keras.Input(shape=(224, 224, 3)))
model.add(layers.Conv2D(filters = 96, kernel_size = [11, 11], strides = [4, 4], padding = "valid", activation = "relu"))
model.add(layers.MaxPooling2D(pool_size = [2, 2], strides = [2, 2], padding = "valid"))

model.add(layers.Conv2D(filters = 256, kernel_size = [5, 5], strides = [1, 1], padding = "valid", activation = "relu"))
model.add(layers.MaxPooling2D(pool_size = [2, 2], strides = [2, 2], padding = "valid"))

model.add(layers.Conv2D(filters = 384, kernel_size = [3, 3], strides = [1, 1], padding = "valid", activation = "relu"))

model.add(layers.Conv2D(filters = 384, kernel_size = [3, 3], strides = [1, 1], padding = "valid", activation = "relu"))

model.add(layers.Conv2D(filters = 256, kernel_size = [3, 3], strides = [1, 1], padding = "valid", activation = "relu"))
model.add(layers.MaxPooling2D(pool_size = [2, 2], strides = [2, 2], padding = "valid"))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(rate = 0.2))

model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(rate = 0.5))

model.add(layers.Dense(200, activation="softmax"))

model.compile(
  loss = "categorical_crossentropy",
  optimizer = keras.optimizers.SGD(momentum = 0.9, decay = 0.0005),
  metrics = ["accuracy"]
)
  
datagen = keras.preprocessing.image.ImageDataGenerator(
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = True,
  zca_whitening = True)

start_time = time.time()
model.fit_generator(
  datagen.flow_from_directory(directory = tiny_imagenet_train,
                              batch_size = 128,
                              target_size = [224, 224]),
  steps_per_epoch = 10
)
print("--- %s seconds ---" % (time.time() - start_time))
