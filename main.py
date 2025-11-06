from CPH2d import ChiralPersistentHomology2D
import numpy as np
import time
import matplotlib.pyplot as plt

dataset = np.random.uniform(-1, 1, (1000, 3))
# plt.scatter(dataset[:, 0], dataset[:, 1], s=0.2)
# plt.savefig('figs/dataset.png', dpi=300, bbox_inches='tight')
# plt.close()

chiral_ph = ChiralPersistentHomology2D(dataset)
bottleneck_distance = chiral_ph.bottleneck_distance()
print(bottleneck_distance)

# wasserstein_distance = chiral_ph.wasserstein_distance()
# print(wasserstein_distance)

# for i in range(10):
#     np.random.seed(i+10)
#     dataset = np.random.uniform(-1, 1, (2000, 3))
#     start_time = time.time()
#     chiral_ph = ChiralPersistentHomology2D(dataset)

#     bottleneck_distance = chiral_ph.bottleneck_distance()
#     print(bottleneck_distance)
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")



