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

# plt.scatter(index, loss, s=20, label="AE koder - strata")
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
        loss.append(float(line)/1000)
        index.append(i)

# plt.scatter(index, loss, s=20, label="VAE koder - strata")
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
        loss.append(float(line)/10)
        index.append(i)

# plt.scatter(index, loss, s=20, label="CGAN generator - strata")
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

# plt.scatter(index, time, s=20, label="GAN time")
# plt.legend()
# plt.show()
# plt.close()

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

# plt.scatter(index, time, s=20, label="AE time")
# plt.legend()
# plt.show()
# plt.close()

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

# plt.scatter(index, time, s=20, label="VAE time")
# plt.legend()
# plt.show()
# plt.close()

# ********************************************************************

plt.bar(["AE"], [78], label='AE')
plt.bar(["VAE"], [110], label='VAE')
plt.bar(["CGAN"], [39], label='GAN')
plt.title("Objętość modeli w [Mb]")
plt.legend()
plt.show()

# ********************************************************************

plt.bar(["AE"], [4], label='AE')
plt.bar(["VAE"], [5], label='VAE')
plt.bar(["CGAN"], [38], label='GAN')
plt.title("Czas uczenia w [h] dla 40 epok")
plt.legend()
plt.show()

# ********************************************************************

plt.bar(["AE"], [40], label='AE')
plt.bar(["VAE"], [40], label='VAE')
plt.bar(["CGAN"], [5], label='GAN')
plt.title("Zadowalające wyniki po [n] epokach")
plt.legend()
plt.show()

# ********************************************************************

plt.bar(["AE"], [4], label='AE')
plt.bar(["VAE"], [4], label='VAE')
plt.bar(["CGAN"], [8], label='GAN')
plt.title("Czas procesowania dla 100 obrazów [s]")
plt.legend()
plt.show()