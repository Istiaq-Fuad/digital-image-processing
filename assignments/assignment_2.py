import matplotlib.pyplot as plt
import numpy as np

colors = np.array(
    [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
)
names = ["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"]


shades = np.zeros((7, 3, 3))


for i in range(7):
    for j in range(3):
        shades[i, j] = colors[i] * (j / 2)


fig, ax = plt.subplots(7, 3, figsize=(6, 10))
for i in range(7):
    for j in range(3):
        image = np.broadcast_to(shades[i, j], (50, 50, 3))
        ax[i, j].imshow(image)
        ax[i, j].set_title(f"{names[i]} {j}")
        ax[i, j].axis("off")

plt.tight_layout()
plt.show()
