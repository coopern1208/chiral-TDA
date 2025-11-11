import numpy as np

def generate_fake_data(n_tri= 1000, base_tri = [[0, 0, 1], [1, 0, 0.7], [0, 2, 0.2]], seed = 42):
    triangles  = np.array(base_tri)[:, :2]
    lumino = np.array(base_tri)[:, 2]
   
    all_points = np.tile(triangles, (n_tri, 1, 1))

    rng = np.random.default_rng(seed=seed)
    scale = rng.uniform(0.04, 0.08, n_tri)
    angles = rng.uniform(0, 2*np.pi, n_tri)
    translation = rng.uniform(-1, 1, (n_tri, 2))

    cos_angles = np.cos(angles)[:, np.newaxis]
    sin_angles = np.sin(angles)[:, np.newaxis]

    all_points = all_points * scale[:, np.newaxis, np.newaxis]
    rotated_points = np.empty_like(all_points)
    rotated_points[:, :, 0] = (
        all_points[:, :, 0] * cos_angles - all_points[:, :, 1] * sin_angles
    )
    rotated_points[:, :, 1] = (
        all_points[:, :, 0] * sin_angles + all_points[:, :, 1] * cos_angles
    )
    all_points = rotated_points
    all_points = all_points + translation[:, np.newaxis, :]
    lumino_expanded = np.tile(lumino, (n_tri, 1))[:, :, np.newaxis] * np.random.uniform(0.8, 1.2, (n_tri, 1))[:, np.newaxis]

    all_points = np.concatenate((all_points, lumino_expanded), axis=2)
    flatten_coords = all_points.reshape(-1, 3)
    return all_points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = generate_fake_data(n_tri= 100)
    print(data.shape)
    print(data[:10])
    for i in data:
        plt.scatter(i[:, 0], i[:, 1], s=1, marker='o', color='black')
        plt.fill(i[:, 0], i[:, 1], color='red', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('fake_data.png')
    plt.close()