import os
import cv2
import numpy as np
from imgaug import augmenters as iaa


goal_len = 400

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-3, 3),
        shear=(-4, 4)
    ),
    iaa.Crop(px = (0, 16)),
    iaa.Multiply((0.8, 1.2), per_channel = 0.4)
])

def process_category(path):
    print(path)

    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    to_create = goal_len - len(files)
    if to_create < 0:
        print("Goal already achieved")
        return
    images = []
    for i in range(to_create):
        derive_from_i = i % len(files)
        derive_from_f = files[derive_from_i]
        #print("f: " + derive_from_f)
        img = cv2.imread(derive_from_f, cv2.IMREAD_COLOR)
        #print("img type " + str(np_img.shape))
        np_img = np.asarray(img)
        images.append(np_img)
    
    images_aug = seq(images = images)
    
    for i, img in enumerate(images_aug):
        save_path = os.path.join(path, "Gen-" + str(i) + ".jpg")
        cv2.imwrite(save_path, img)

        


for category in os.scandir(os.path.dirname(__file__)):
    if not category.is_dir():
        continue
    if category.name.startswith("."):
        continue
    process_category(category.path)



