import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "depth_75.png"
im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
img_np = np.array(im, dtype=np.float16)
print(img_np)
print(img_np.dtype)
print(im)
print(im.dtype)
num_zeros = 0
num_large = 0
total = 0
large_x = []
large_y = []
other_x = []
other_y = []
zeroz_x = []
zeroz_y = []
max_val = 0
for i, row in enumerate(im):
    for j, pixel in enumerate(row):
        total += 1
        if pixel == 0:
            num_zeros += 1
            zeroz_x.append(i)
            zeroz_y.append(j)
        if pixel == 65535:
            num_large += 1
            large_x.append(i)
            large_y.append(j)
        else:
            if pixel > max_val:
                max_val = pixel
            other_x.append(i)
            other_y.append(j)

plt.scatter(zeroz_x, zeroz_y, s=1)
plt.scatter(large_x, large_y, s=1)
plt.scatter(other_x, other_y, s=1)
plt.legend(["Zero", "Large", "Other"])
plt.savefig("large_values.png")

print("next largest val: ", max_val)

print(total)
print(num_zeros)
print(num_large)
print(im.max())
print(num_zeros/total)
print(num_large/total)