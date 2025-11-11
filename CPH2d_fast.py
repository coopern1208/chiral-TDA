#from typing import List, Dict, Tuple, Optional, Union, Any

from typing import Any
import numpy as np
import gudhi as gd

import matplotlib.pyplot as plt


class ChiralPersistentHomology2D:
    def __init__(self, dataset: np.ndarray, max_alpha: float = 0.05):
        # (x,y,luminosity) with shape (num_galaxies, 3)
        self.dataset: np.ndarray = dataset
        self.num_galaxies: int = dataset.shape[0]
        self.data_locations: np.ndarray = self.dataset[:, :2]

        self.max_alpha: float = max_alpha

    def _randomize_luminosity(self, seed: int = 42):
        # randomize the luminosity of the dataset
        rng = np.random.default_rng(seed=seed)
        randomized_dataset = self.dataset.copy()
        randomized_dataset[:, 2] = rng.random(self.num_galaxies)
        return randomized_dataset

    def _compute_alpha_complex(self):
        alpha_complex = gd.AlphaComplex(points=self.dataset[:, :2])
        simplex_tree = alpha_complex.create_simplex_tree()
        
        # ---- edges ---- #
        edges_and_alpha = [list[int, float](s[0]) + [s[1]] 
                            for s in simplex_tree.get_skeleton(1) 
                            if len(s[0]) == 2 and s[1] <= self.max_alpha
                          ]
        self.E = np.array(edges_and_alpha, dtype=int)[:,:2]
        self.alpha_E = np.array(edges_and_alpha, dtype=float)[:,2]

        # ---- triangles ---- #
        triangles_and_alpha = [
            # sort the triangles by the luminosities of the galaxies
            (np.array(s[0])[np.argsort(self.dataset[s[0]][:,2])].tolist() + [s[1]]) 
            for s in simplex_tree.get_skeleton(2)
            if len(s[0]) == 3 and s[1] <= self.max_alpha
        ]
        self.T = np.array(triangles_and_alpha, dtype=int)[:,:3]
        self.alpha_T = np.array(triangles_and_alpha, dtype=float)[:,3]

    def _split_chiral_triangles(self):
        triangle_coords = np.array([[self.data_locations[tri[0]], self.data_locations[tri[1]], self.data_locations[tri[2]]] for tri in self.T])

        ab = triangle_coords[:, 1] - triangle_coords[:, 0]
        ac = triangle_coords[:, 2] - triangle_coords[:, 0]
        oriented_area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        
        is_right_handed = oriented_area > 0

        # ---- triangles ---- #
        self.RH_T = self.T[is_right_handed]
        self.RH_alpha_T = self.alpha_T[is_right_handed]
        self.LH_T = self.T[~is_right_handed]
        self.LH_alpha_T = self.alpha_T[~is_right_handed]


        # ---- edges ---- #
        def extract_edges(triangles):
            edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]])
            edges = np.sort(edges, axis=1)
            unique_edges = np.unique(edges, axis=0)
            return unique_edges

        self.LH_E = extract_edges(self.LH_T)
        self.RH_E = extract_edges(self.RH_T)

        self.boundary_E = np.unique(np.vstack([self.LH_E, self.RH_E]), axis=0)
        self.free_E = np.array([e for e in self.E if e not in self.boundary_E])
        #print(self.free_E)

        def find_index(A, B):
            if len(A) == 0 or len(B) == 0:
                return []
            A_view = A.view([('', A.dtype)] * A.shape[1])
            B_view = B.view([('', B.dtype)] * B.shape[1])
            indices = np.nonzero(A_view == B_view[:, None])[1]
            return indices

        self.LH_alpha_E = self.alpha_E[find_index(self.LH_E, self.E)]
        self.RH_alpha_E = self.alpha_E[find_index(self.RH_E, self.E)]
        self.free_alpha_E = self.alpha_E[find_index(self.free_E, self.E)]

    def _chiral_complex_tree(self):
        LH_complex = gd.SimplexTree()
        RH_complex = gd.SimplexTree()

        for edge, alpha in zip(self.boundary_E, self.LH_alpha_E):
            LH_complex.insert(edge, filtration=alpha)
        for edge, alpha in zip(self.free_E, self.free_alpha_E):
            LH_complex.insert(edge, filtration=alpha)
        for triangle, alpha in zip(self.LH_T, self.LH_alpha_T):
            LH_complex.insert(triangle, filtration=alpha)

        for edge, alpha in zip(self.boundary_E, self.RH_alpha_E):
            RH_complex.insert(edge, filtration=alpha)
        for triangle, alpha in zip(self.RH_T, self.RH_alpha_T):
            RH_complex.insert(triangle, filtration=alpha)   
        for edge, alpha in zip(self.free_E, self.free_alpha_E):
            RH_complex.insert(edge, filtration=alpha)
        return LH_complex, RH_complex

    def chiral_persistence(self):
        self._compute_alpha_complex()
        self._split_chiral_triangles()
        LH_complex, RH_complex = self._chiral_complex_tree()

        LH_dgm = LH_complex.persistence()
        LH_dgm = [(dim, (birth, death)) for dim, (birth, death) in LH_dgm if death != float('inf')]
        RH_dgm = RH_complex.persistence()
        RH_dgm = [(dim, (birth, death)) for dim, (birth, death) in RH_dgm if death != float('inf')]
        
        return LH_dgm, RH_dgm


    def plot_chiral_persistence(self):
        LH_dgm, RH_dgm = self.chiral_persistence()

        gd.plot_persistence_diagram(LH_dgm, legend = True)
        plt.xlim(-0.0003, 0.01)
        plt.ylim(-0.0003, 0.01)
        plt.savefig('figs/LH_dgm_fast.png', dpi=300)
        plt.close()

        RH_dgm = [(dim, (birth, death)) for dim, (birth, death) in RH_dgm if death != float('inf')]
        gd.plot_persistence_diagram(RH_dgm, legend = True)
        plt.xlim(-0.0003, 0.01)
        plt.ylim(-0.0003, 0.01)
        plt.savefig('figs/RH_dgm_fast.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    dataset = np.random.uniform(-1, 1, (1000, 3))

    chiral_ph = ChiralPersistentHomology2D(dataset, max_alpha=0.05)
    chiral_ph.plot_chiral_persistence()