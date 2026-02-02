import os
from torchvision import datasets

RAW_DIR = "./Data/raw"
OUTPUT_DIR = "./Data/mnist"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dataset = datasets.MNIST(root=RAW_DIR, train=True, download=True)
test_dataset = datasets.MNIST(root=RAW_DIR, train=False, download=True)

def save_dataset(dataset, split):
    for idx, (image, label) in enumerate(dataset):
        class_dir = os.path.join(OUTPUT_DIR, split, str(label))
        os.makedirs(class_dir, exist_ok=True)
        image.save(os.path.join(class_dir, f"{idx}.jpg"))

    print(f"{split} saved!")

save_dataset(train_dataset, "train")
save_dataset(test_dataset, "test")

print("Done!")