import numpy as np
import matplotlib.pyplot as plt


path = '/home/nemodrive/workspace/andreim/self_supervised_steering_results/hard/01fd5e96d7134f50/frame000012.npy'

arr = np.load(path)
print(arr)
plt.bar(arr[0], arr[1])
plt.show()