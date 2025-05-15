import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import kagglehub
import os
import shutil



(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)

path = kagglehub.dataset_download("jcprogjava/handwritten-digits-dataset-not-in-mnist")
print("Path to dataset files:", path)

source_dir = os.path.join(path, "dataset")
target_dir = os.path.join(path, "flattened_digits")

os.makedirs(target_dir, exist_ok=True)

for digit in range(10):
    src = os.path.join(source_dir, str(digit), str(digit))
    dst = os.path.join(target_dir, str(digit))
    os.makedirs(dst, exist_ok=True)

    if os.listdir(dst):
        continue

    for img_file in os.listdir(src):
        full_src = os.path.join(src, img_file)
        full_dst = os.path.join(dst, img_file)
        shutil.copy(full_src, full_dst)


ds_kaggle_test = tf.keras.utils.image_dataset_from_directory(
    os.path.join(path, "flattened_digits"),
    labels='inferred',
    label_mode='int',
    image_size=(28, 28),
    color_mode='grayscale',
    batch_size=128,
    shuffle=True 
)

ds_kaggle_test = ds_kaggle_test.map(normalize_img).cache().prefetch(tf.data.AUTOTUNE)

#for images, labels in ds_kaggle_test.take(1):
#    print("Sample labels:", labels.numpy())

model.evaluate(ds_kaggle_test)