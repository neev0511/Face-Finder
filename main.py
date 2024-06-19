from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import shutil


def moveImg(path):
    shutil.copyfile(os.path.join(all_images_path, path),
                    os.path.join(personal_images_path, path))


reference_image_path = input("Reference Image Path: ")
all_images_path = input("All Images Path: ")
personal_images_path = input("Same Person Folder Path: ")

img1 = cv2.imread(reference_image_path)
plt.imshow(img1[:, :, ::-1])


for filename in os.listdir(all_images_path):
    img_path = os.path.join(all_images_path, filename)

    img2 = cv2.imread(img_path)
    plt.imshow(img2[:, :, ::-1])

    result = DeepFace.verify(img1, img2, enforce_detection=False)
    if result['verified']:
        moveImg(filename)
