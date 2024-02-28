import os
from PIL import Image
import numpy as np
from face_recognition import face_locations

data = []

for root, _, files in os.walk("./Dataset"):
    for file in files:
        path = os.path.join(root, file)
        if path in data:
            break
        data.append(path)
        print(path)
        img = np.asarray(Image.open(path).convert("RGB"))
        image_array = np.array(img, "uint8")
        try:
            (top, right, bottom, left) = face_locations(img)[0]
            image_array = image_array[top:bottom, left:right]
            img = Image.fromarray(image_array)
            img = img.resize((112, 112))
            img.save(path)
        except Exception as e:
            os.remove(path)

