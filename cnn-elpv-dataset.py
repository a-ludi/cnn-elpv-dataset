#!/usr/bin/env python3
"""CNN-based identification of defective solar cells in electroluminescence imagery.

@author: Arne Ludwig <arne.ludwig@posteo.de>
@copyright: Copyright © 2022 by Arne Ludwig
"""

import matplotlib.pyplot as plt

from elpv_dataset.utils.elpv_reader import load_dataset

# %% Load data

images, probs, types = load_dataset()


# %% Visualize data

n_samples = 3
plt.figure(figsize=(6.4, n_samples * 3.2))
k = 0
for i, type_ in enumerate(["mono"] * n_samples + ["poly"] * n_samples):
    for j in range(4):
        prob = j / 3
        sample = images[(probs == prob) & (types == type_)][i % n_samples]
        k += 1
        plt.subplot(2 * n_samples, 4, k)
        plt.imshow(sample)
        plt.axis("image")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title(f"p = {j}/3")
        if j == 0 and i % n_samples == 0:
            plt.ylabel(f"← {type_}")
plt.tight_layout()
plt.show()
