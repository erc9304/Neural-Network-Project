import tensorflow as tf
import tensorflow_datasets as tfds
import time

#ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
#tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
#tfds.benchmark(ds, batch_size=32)

class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
            args=(num_samples,)
        )
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)

benchmark(ArtificialDataset())