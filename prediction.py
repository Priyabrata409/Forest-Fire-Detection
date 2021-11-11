from math import pi
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import time
model = load_model("saved_model\model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
cap = cv2.VideoCapture(
    r"C:\Users\priyabrata\Forest_Fire_Detection\data\videos\no_fire_video.mp4"
)
# generating random data values
x = np.array([1, 2])
y = np.array([2, 3])

# enable interactive mode
plt.ion()
# creating subplot and figure
fig = plt.figure()
ax = fig.add_subplot(111)
bar = ax.bar(x, y)
s=time.time()
# setting labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Updating plot...")
f=0
# looping
while cap.isOpened():
    if f-s > 0:
        print(" The frame rate is ", np.round(1/(f-s),2),"fps")
    _, frame = cap.read()
    cv2.imshow("frame", frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.reshape(1, 320, 320, 3)
    img = img / 255.0
    res = model.predict(img)
    res = res.flatten()[0]
    # updating the value of x and y
    bar.remove()
    y = np.array([1 - res, res])
    bar = ax.bar(
        x, y, tick_label=["No Fire", "Fire"], width=0.5, color=["green", "red"]
    )
    ax.set_ylim([0, 1.5])

    # re-drawing the figure
    fig.canvas.draw()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # to flush the GUI events
    s=f
    f=time.time()
    fig.canvas.flush_events()
    time.sleep(0.0001)


cap.release()
cv2.destroyAllWindows()
