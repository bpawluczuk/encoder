from datetime import datetime
import numpy
import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

# ********************************************************************
dir = "history/AE/"

loss = []
index = []
i = 0
with open(dir + "stats_loss.txt", "r") as f:
    for line in f:
        i = i + 1
        loss.append(float(line))
        index.append(i)

# plt.scatter(index, loss, s=20, label="AE koder loss")
# plt.legend()
# plt.show()
# plt.close()

# ********************************************************************
dir = "history/VAE/"

loss = []
index = []
i = 0
with open(dir + "stats_loss.txt", "r") as f:
    for line in f:
        i = i + 1
        loss.append(float(line))
        index.append(i)

# plt.scatter(index, loss, s=20, label="VAE koder loss")
# plt.legend()
# plt.show()
# plt.close()

# ********************************************************************
dir = "history/GAN/"

loss = []
index = []
i = 0
with open(dir + "stats_loss_gen.txt", "r") as f:
    for line in f:
        i = i + 1
        loss.append(float(line))
        index.append(i)

# plt.scatter(index, loss, s=20, label="GAN generator loss")
# plt.legend()
# plt.show()
# plt.close()

# ********************************************************************
dir = "history/GAN/"

time = []
time_epoc = 0
index = []
i = 0
with open(dir + "stats_time.txt", "r") as f:
    for line in f:
        i = i + 1
        date_time_obj = datetime.strptime(str(line).rstrip('\n'), '%H:%M:%S.%f')
        time_epoc = time_epoc + date_time_obj.minute
        time.append(str(round(time_epoc / 60, 2)))
        index.append(i)

plt.scatter(index, time, s=20, label="GAN time")
plt.legend()
plt.show()
plt.close()

# ********************************************************************
dir = "history/AE/"

time = []
time_epoc = 0
index = []
i = 0
with open(dir + "stats_time.txt", "r") as f:
    for line in f:
        i = i + 1
        date_time_obj = datetime.strptime(str(line).rstrip('\n'), '%H:%M:%S.%f')
        time_epoc = time_epoc + date_time_obj.minute
        time.append(str(round(time_epoc / 60, 2)))
        index.append(i)

plt.scatter(index, time, s=20, label="AE time")
plt.legend()
plt.show()
plt.close()

# ********************************************************************
dir = "history/VAE/"

time = []
time_epoc = 0
index = []
i = 0
with open(dir + "stats_time.txt", "r") as f:
    for line in f:
        i = i + 1
        date_time_obj = datetime.strptime(str(line).rstrip('\n'), '%H:%M:%S.%f')
        time_epoc = time_epoc + date_time_obj.minute
        time.append(str(round(time_epoc / 60, 2)))
        index.append(i)

plt.scatter(index, time, s=20, label="VAE time")
plt.legend()
plt.show()
plt.close()