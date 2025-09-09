import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt


class ChiralPersistentHomology3D:
    def __init__(self, dataset, max_alpha = 0.05):
        self.dataset = dataset
        self.max_alpha = max_alpha

    def compute_alpha_complex(self):
        alpha_complex = gd.AlphaComplex(points=self.dataset)
        simplex_tree = alpha_complex.create_simplex_tree()
        return simplex_tree

