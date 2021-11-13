import datetime
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

plt.scatter(index, loss, s=20, label="Encoder loss")
plt.legend()
plt.show()
plt.close()
