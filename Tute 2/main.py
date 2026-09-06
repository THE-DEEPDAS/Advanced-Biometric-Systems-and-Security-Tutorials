import os

dataset_path = "face_dataset"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            count += 1

print("Total images:", count)