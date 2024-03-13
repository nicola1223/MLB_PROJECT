import face_recognition
import numpy as np
import torch
from datasets import WorkDataset
from network import Encoder, Decoder, Autoencoder, ConvolutionalAutoencoder
import cv2
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda:0")
model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder(), device))

res_dataset = WorkDataset("/home/nicola/TheBestProject/src/network/Dataset")

model.load_model("../autoencoder_weights.pth")

with torch.no_grad():
    image_features = [model.encode(x[0].to(device)) for x in tqdm(res_dataset)]

img = np.asarray(Image.open("../../../TheBestProject/src/network/Unit_tests_images/image3.jpg"))

(top, right, bottom, left) = face_recognition.face_locations(img)[0]

face_img = img[top:bottom, left:right]
resized_image = cv2.resize(face_img, (112, 112))
image = np.array(resized_image, "uint8")

most_similar = model.visual_similarity(image, res_dataset, image_features)

print(len(most_similar))

for i, x in enumerate(most_similar):
    print(f"{x[1]} - {x[2]}")
    cv2.imshow(f"Like {i+1}", x[0][:, :, ::-1])

cv2.imshow("Original", image[:, :, ::-1])

cv2.waitKey(0)

cv2.destroyAllWindows()
