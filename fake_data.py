import numpy as np
import matplotlib.pyplot as plt

class FakeData2D:
    def __init__(self, n_tri= 1000, base_tri = [[0, 0, 1], [1, 0, 0.7], [0, 2, 0.2]]):
        self.n_tri = n_tri
        self.base_tri = base_tri


        
 

    def plot_example_triangle(self):
        plt.scatter(self.example_xy[:, 0], self.example_xy[:, 1], c=self.example_luminosity, s=1, marker='o')
        plt.savefig('example_triangle.png')
        plt.close()

    def oriented_area(self, triangle):
        ab = triangle[1] - triangle[0]
        ac = triangle[2] - triangle[0]
        return 0.5 * (ab[0] * ac[1] - ab[1] * ac[0])

    def _transform_triangle(self, triangle, angle, scale, translation):
        x = triangle[:, 0]
        y = triangle[:, 1]
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        x_new = x_new * scale + translation[0]
        y_new = y_new * scale + translation[1]
        return np.column_stack((x_new, y_new))

    def test(self):
        triangle = self.example_triangle[:,:2]
        print(triangle)

if __name__ == "__main__":
    fake_data = FakeData2D()
    fake_data.all_points