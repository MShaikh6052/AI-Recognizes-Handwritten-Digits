import tensorflow
import numpy
import cv2
import os
import matplotlib.pyplot as plt
from train import Train

train = Train()
train.build()

nn_model = tensorflow.keras.models.load_model("nn.model")

for image in os.listdir("h_digits"):
    digit = cv2.imread(f"h_digits/{image}")[:,:,0]
    digit = numpy.invert(numpy.array([digit]))
    ans = nn_model.predict(digit)
    print(f"This digit is {numpy.argmax(ans)}")
    plt.imshow(digit[0], cmap=plt.cm.binary)
    plt.show()
