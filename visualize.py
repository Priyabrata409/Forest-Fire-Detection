from temperature_simulator import get_weather_data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import collections


def my_function(i):
    # get data
    temp, pressure, humidity = get_weather_data()
    t.popleft()
    t.append(temp)
    p.popleft()
    p.append(pressure)
    h.popleft()
    h.append(humidity)
    # clear axis
    ax.cla()
    ax1.cla()
    ax2.cla()
    # plot temperature
    ax.plot(t)
    ax.scatter(len(t) - 1, t[-1])
    ax.text(len(t) - 1, t[-1] + 2, f"{np.round(t[-1],2)}C")
    ax.set_ylim(0, 100)
    ax.set_title("Temperature")
    # plot pressure
    ax1.plot(p)
    ax1.scatter(len(p) - 1, p[-1])
    ax1.text(len(p) - 1, p[-1] + 2, f"{np.round(p[-1],2)}hPa")
    ax1.set_ylim(1000, 1200)
    ax1.set_title("Pressure")
    # Plot humidity
    ax2.plot(h)
    ax1.scatter(len(h) - 1, h[-1])
    ax2.text(len(h) - 1, h[-1] + 2, f"{np.round(h[-1],2)}%")
    ax2.set_ylim(50, 100)
    ax2.set_title("Humidity")


# start collections with zeros
t = collections.deque(np.zeros(10))
h = collections.deque(np.zeros(10))
p = collections.deque(np.zeros(10))
# define and adjust figure
fig = plt.figure(figsize=(18, 6), facecolor="#DEDEDE")
ax = plt.subplot(131)
ax1 = plt.subplot(132)
ax2 = plt.subplot(133)
ax.set_facecolor("#DEDEFF")
ax1.set_facecolor("#DEFEFA")
ax2.set_facecolor("#DEDEAE")
# animate
ani = FuncAnimation(fig, my_function, interval=1000)
plt.show()
