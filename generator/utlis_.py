import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, granularity) -> None:
        self.granularity = granularity
        self.dimension = int(1/granularity)
        self.matrix = np.zeros((self.dimension, self.dimension))

    def sum_matrices_with_center(self, matrix_large, matrix_small, center_large):

        rows_large, cols_large = matrix_large.shape
        rows_small, cols_small = matrix_small.shape

        start_row = center_large[0] - rows_small // 2
        start_col = center_large[1] - cols_small // 2

        for i in range(rows_small):
            for j in range(cols_small):
                row_large = start_row + i
                col_large = start_col + j

                if 0 <= row_large < rows_large and 0 <= col_large < cols_large:
                    matrix_large[row_large, col_large] += matrix_small[i, j]

        return matrix_large

    def generateGaussian(self, img, cx, cy, sigma=3):
        size = 6*sigma + 3

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return self.sum_matrices_with_center(img, g, (cy, cx))

    def generateDistribution(self, row):

        for i in range(1, 8):
            if pd.isna(row[f"x_pos_{i}"]) or pd.isna(row[f"y_pos_{i}"]):
                continue
            x = int(row[f"x_pos_{i}"]//self.granularity)
            y = int(row[f"y_pos_{i}"]//self.granularity)
            self.matrix = self.generateGaussian(
                self.matrix, x, y, self.dimension//50)

    def plot(arg):
        mat = arg.matrix if isinstance(arg, Distribution) else arg

        plt.imshow(mat, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.title('Distribution')
        plt.show()
