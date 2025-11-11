from typing import Any


import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt


class ChiralPersistentHomology2D:
    def __init__(self, dataset, max_alpha = 0.05):
        self.dataset = dataset
        self.max_alpha = max_alpha

    def compute_alpha_complex(self):
        alpha_complex = gd.AlphaComplex(points=self.dataset[:, :2])
        simplex_tree = alpha_complex.create_simplex_tree()
        
        edges_and_alpha = [list(s[0]) + [s[1]] for s in simplex_tree.get_skeleton(1) if len(s[0]) == 2]
        edges = np.array(edges_and_alpha, dtype=int)[:,:2].tolist()
        edges = [tuple(sorted(edge)) for edge in edges]
        edges_alpha = np.array(edges_and_alpha)[:,2].tolist()

        triangles_and_alpha = [
            (np.array(s[0])[np.argsort(self.dataset[s[0]][:,2])].tolist() + [s[1]]) # sort the triangles by the point values
            for s in simplex_tree.get_skeleton(2)
            if len(s[0]) == 3 and s[1] <= self.max_alpha
        ]
        triangles = np.array(triangles_and_alpha, dtype=int)[:,:3]
        triangles_alpha = np.array(triangles_and_alpha)[:,3]

        return edges, edges_alpha, triangles, triangles_alpha
        
    def chiral_alpha_complex(self):
        edges, edges_alpha, triangles, triangles_alpha = self.compute_alpha_complex()

        # ---- triangles ---- #
        triangle_coords = np.array([[self.dataset[tri[0]], self.dataset[tri[1]], self.dataset[tri[2]]] for tri in triangles])
        ab = triangle_coords[:, 1] - triangle_coords[:, 0]
        ac = triangle_coords[:, 2] - triangle_coords[:, 0]
        oriented_area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]

        is_right_handed = oriented_area > 0

        RH_triangles = triangles[is_right_handed]
        RH_triangles_alpha = triangles_alpha[is_right_handed]
        LH_triangles = triangles[~is_right_handed]
        LH_triangles_alpha = triangles_alpha[~is_right_handed]
        
        # ---- edges ---- #
        LH_boundary_edges = []
        RH_boundary_edges = []

        def extract_edges(triangles):
            # triangles: (n, 3) array
            edges_array = np.concatenate([
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [0, 2]]
            ], axis=0)
            return list(set([tuple(sorted(edge)) for edge in edges_array]))

        LH_boundary_edges = extract_edges(LH_triangles)
        RH_boundary_edges = extract_edges(RH_triangles)

        boundary_edges = list[tuple[Any, ...]](set(LH_boundary_edges + RH_boundary_edges))
        free_edges = [e for e in edges if tuple(e) not in boundary_edges]

        LH_boundary_edges_alpha = [edges_alpha[edges.index(edge)] for edge in LH_boundary_edges]
        RH_boundary_edges_alpha = [edges_alpha[edges.index(edge)] for edge in RH_boundary_edges]
        free_edges_alpha = [edges_alpha[edges.index(edge)] for edge in free_edges]

        LH_complex = gd.SimplexTree()
        RH_complex = gd.SimplexTree()

        for edge, alpha in zip(boundary_edges, LH_boundary_edges_alpha):
            LH_complex.insert(edge, filtration=alpha)
        for edge, alpha in zip(free_edges, free_edges_alpha):
            LH_complex.insert(edge, filtration=alpha)
        for triangle, alpha in zip(LH_triangles, LH_triangles_alpha):
            LH_complex.insert(triangle, filtration=alpha)

        for edge, alpha in zip(boundary_edges, RH_boundary_edges_alpha):
            RH_complex.insert(edge, filtration=alpha)
        for edge, alpha in zip(free_edges, free_edges_alpha):
            RH_complex.insert(edge, filtration=alpha)
        for triangle, alpha in zip(RH_triangles, RH_triangles_alpha):
            RH_complex.insert(triangle, filtration=alpha)

        return LH_complex, RH_complex

    def chiral_persistence(self):
        LH_complex, RH_complex = self.chiral_alpha_complex()
        LH_dgm = LH_complex.persistence()
        LH_dgm = [(dim, (birth, death)) for dim, (birth, death) in LH_dgm if death != float('inf')]
        RH_dgm = RH_complex.persistence()
        RH_dgm = [(dim, (birth, death)) for dim, (birth, death) in RH_dgm if death != float('inf')]
        return LH_dgm, RH_dgm

    # ------------------------------------------------------------ #
    # Plotting functions
    # ------------------------------------------------------------ #
    def plot_chiral_alpha_complex(self):
        edges, edges_alpha, triangles, triangles_alpha = self.compute_alpha_complex()
        triangle_coords = np.array([[self.dataset[tri[0]], self.dataset[tri[1]], self.dataset[tri[2]]] for tri in triangles])
        ab = triangle_coords[:, 1] - triangle_coords[:, 0]
        ac = triangle_coords[:, 2] - triangle_coords[:, 0]
        #print(ab.shape, ac.shape)
        oriented_area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]

        is_right_handed = oriented_area > 0
        RH_triangles = triangles[is_right_handed]
        LH_triangles = triangles[~is_right_handed]
        
        # Create a single figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Helper function to plot triangles
        def plot_triangles(ax, triangles, color, title):
            ax.scatter(self.dataset[:, 0], self.dataset[:, 1], c='black', marker='o', s=1, alpha=0.6)
            for simplex in triangles:
                if len(simplex) == 3:
                    a = self.dataset[simplex[0]]
                    b = self.dataset[simplex[1]]
                    c = self.dataset[simplex[2]]
                    ax.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=0.2)
                    ax.plot([b[0], c[0]], [b[1], c[1]], color='black', linewidth=0.2)
                    ax.plot([c[0], a[0]], [c[1], a[1]], color='black', linewidth=0.2)
                    ax.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color=color, alpha=0.3)
            ax.set_title(title)
            ax.set_aspect('equal')
        
        # Plot right-handed triangles
        plot_triangles(axes[0], RH_triangles, 'blue', 'Right-handed Triangles')
        
        # Plot left-handed triangles
        plot_triangles(axes[1], LH_triangles, 'red', 'Left-handed Triangles')
        
        # Plot combined alpha complex
        axes[2].scatter(self.dataset[:, 0], self.dataset[:, 1], c='black', marker='o', s=1, alpha=0.6)
        
        # Plot all edges
        for edge in edges:
            a = self.dataset[edge[0]]
            b = self.dataset[edge[1]]
            axes[2].plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=0.2)
        
        # Plot left-handed triangles
        for tri in LH_triangles:
            a = self.dataset[tri[0]]
            b = self.dataset[tri[1]]
            c = self.dataset[tri[2]]
            axes[2].fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color='red', alpha=0.3)
        
        # Plot right-handed triangles
        for tri in RH_triangles:
            a = self.dataset[tri[0]]
            b = self.dataset[tri[1]]
            c = self.dataset[tri[2]]
            axes[2].fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color='blue', alpha=0.3)
        
        axes[2].set_title('Alpha Complex')
        axes[2].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('figs/chiral_alpha_complex.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_chiral_persistence(self):
        LH_dgm, RH_dgm = self.chiral_persistence()

        gd.plot_persistence_diagram(LH_dgm, legend = True)
        plt.xlim(-0.0003, 0.01)
        plt.ylim(-0.0003, 0.01)
        plt.savefig('figs/LH_dgm_slow.png', dpi=300)
        plt.close()

        RH_dgm = [(dim, (birth, death)) for dim, (birth, death) in RH_dgm if death != float('inf')]
        gd.plot_persistence_diagram(RH_dgm, legend = True)
        plt.xlim(-0.0003, 0.01)
        plt.ylim(-0.0003, 0.01)
        plt.savefig('figs/RH_dgm_slow.png', dpi=300)
        plt.close()

    def bottleneck_distance(self):
        LH_dgm, RH_dgm = self.chiral_persistence()
        dim0_LH_dgm = [x[1] for x in LH_dgm if x[0] == 0]
        dim0_RH_dgm = [x[1] for x in RH_dgm if x[0] == 0]
        dim0_bottleneck_distance = gd.bottleneck_distance(dim0_LH_dgm, dim0_RH_dgm)

        dim1_LH_dgm = [x[1] for x in LH_dgm if x[0] == 1]
        dim1_RH_dgm = [x[1] for x in RH_dgm if x[0] == 1]
        dim1_bottleneck_distance = gd.bottleneck_distance(dim1_LH_dgm, dim1_RH_dgm)

        return {0: dim0_bottleneck_distance, 1: dim1_bottleneck_distance}

    def wasserstein_distance(self, q=1.0):
        import gudhi.wasserstein as ws
        LH_dgm, RH_dgm = self.chiral_persistence()

        def extract_diagram(persistence, dimension):
            diagram = [[birth, death] for dim, (birth, death) in persistence if dim == dimension]
            # Convert to NumPy array - this is the key fix!
            return np.array(diagram) if diagram else np.empty((0, 2))

        dim0_LH_dgm = extract_diagram(LH_dgm, 0)
        dim0_RH_dgm = extract_diagram(RH_dgm, 0)
        dim0_wasserstein_distance = ws.wasserstein_distance(dim0_LH_dgm, dim0_RH_dgm, order=q)

        dim1_LH_dgm = extract_diagram(LH_dgm, 1)
        dim1_RH_dgm = extract_diagram(RH_dgm, 1)
        dim1_wasserstein_distance = ws.wasserstein_distance(dim1_LH_dgm, dim1_RH_dgm, order=q)
        return {0: dim0_wasserstein_distance, 1: dim1_wasserstein_distance}

if __name__ == "__main__":
    np.random.seed(42)
    dataset = np.random.uniform(-1, 1, (1000, 3))
    chiral_ph = ChiralPersistentHomology2D(dataset, max_alpha=0.05)
    chiral_ph.plot_chiral_persistence()