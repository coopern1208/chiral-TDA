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