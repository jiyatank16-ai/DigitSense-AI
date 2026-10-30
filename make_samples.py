import os
from PIL import Image
from tensorflow.keras.datasets import mnist

os.makedirs("sample_images", exist_ok=True)

print("Downloading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

count = {i: 0 for i in range(10)}

for image, label in zip(x_train, y_train):
    label = int(label)

    if count[label] < 10:
        filename = f"sample_images/digit_{label}_{count[label]+1}.png"
        Image.fromarray(image).save(filename)
        count[label] += 1

    if all(v == 10 for v in count.values()):
        break

print("Successfully created 100 images!")
print("Open sample_images folder.")
