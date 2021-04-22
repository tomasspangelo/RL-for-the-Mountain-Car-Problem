import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class TileCoding:
    """
    Class for the tile coding. One tile coding has multiple tilings.
    """

    def __init__(self, num_tilings, partitions, x_range, y_range, extra_lengths, offset_percent):
        self.tilings = []
        self.num_tilings = num_tilings
        self.partitions = partitions
        self.x_range = x_range
        self.y_range = y_range
        self.extra_lengths = extra_lengths
        self.offset_percent = offset_percent
        self._add_tilings()

    def _add_tilings(self):
        max_offset_x = self.extra_lengths[0] * self.offset_percent
        max_offset_y = self.extra_lengths[1] * self.offset_percent
        for i in range(self.num_tilings):
            offset_x = np.random.uniform(0, max_offset_x)
            offset_y = np.random.uniform(0, max_offset_y)

            tiling = Tiling(self.partitions, self.x_range, self.y_range, offset_x, offset_y)
            self.tilings.append(tiling)

    def get_encoding(self, x, y):

        encoding = np.array([])
        for tiling in self.tilings:
            encoding = np.concatenate(encoding, tiling.get_encoding(x, y))

        return encoding

    def visualize(self):
        fig, ax = plt.subplots()
        ax.plot([self.x_range[0], self.x_range[1]],
                [self.y_range[0], self.y_range[1]],
                alpha=0)
        for tile in self.tilings:
            x = tile.x_range[0] + tile.offset_x
            y = tile.y_range[0] + tile.offset_y
            width = tile.width * self.partitions
            height = tile.height * self.partitions
            ax.add_patch(Rectangle((x, y), width, height,
                                   edgecolor='pink',
                                   facecolor='blue',
                                   fill=True,
                                   lw=5))

        plt.show()


class Tiling:
    """
    Class for a single tiling. Has an offset.
    """

    def __init__(self, partitions, x_range, y_range, offset_x, offset_y):
        self.partitions = partitions
        self.x_range = x_range
        self.y_range = y_range
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.width = (x_range[1] - x_range[0]) / partitions
        self.height = (y_range[1] - y_range[0]) / partitions

    def get_encoding(self, x, y):
        x_low = self.x_range[0] + self.offset_x
        y_low = self.y_range[0] + self.offset_y

        encoding = np.zeros((self.partitions, self.partitions))

        for i in range(self.partitions - 1, -1, -1):
            for j in range(self.partitions):
                x_range = (x_low, x_low + self.width)
                y_range = (y_low, y_low + self.height)
                if self._in_square(x, y, x_range, y_range):
                    encoding[i, j] = 1
                    return encoding.flatten()
                x_low += self.width
            y_low += self.height
        raise ValueError("The point is outside the tiling")

    @staticmethod
    def _in_square(x, y, x_range, y_range):
        if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]:
            return True
        return False


if __name__ == "__main__":
    tc = TileCoding(4, 4, (-1.2, 0.6), (-0.7, 0.7), (0.5, 0.5), 0.5)
    tc.visualize()
