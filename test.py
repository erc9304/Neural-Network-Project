source_dir = os.path.join(path, "dataset")
target_dir = os.path.join(path, "flattened_digits")

os.makedirs(target_dir, exist_ok=True)

for digit in range(10):
    src = os.path.join(source_dir, str(digit), str(digit))
    dst = os.path.join(target_dir, str(digit))
    os.makedirs(dst, exist_ok=True)

    for img_file in os.listdir(src):
        full_src = os.path.join(src, img_file)
        full_dst = os.path.join(dst, img_file)
        shutil.copy(full_src, full_dst)




folders = [os.path.join(path, "dataset", str(i), str(i)) for i in range(10)]
#folders2 = [os.path.join(folder1, str(i), str(i)) for i in range(10)]

ds_kaggle_test = tf.data.Dataset.sample_from_datasets([
    tf.keras.utils.image_dataset_from_directory(
        folder,
        labels='inferred',
        label_mode='int',
        image_size=(28, 28),
        color_mode='grayscale',
        batch_size=128
    ) for folder in folders
])


#ds_kaggle_test = tf.keras.utils.image_dataset_from_directory(
#    image_folder,
#    labels='inferred',
#    label_mode='int',
#    image_size=(28, 28),
#    color_mode='grayscale',
#    batch_size=128,
#    shuffle=False
#)