import numpy as np
import matplotlib.pyplot as plt

class Bin:
    def __init__(self, r=1.0, bin_nums=100, center=np.array([0, 0, 0])):
        """
        Initialize a Bin object to count points on a sphere.

        Parameters:
            r: float, radius of the sphere
            bin_nums: int, number of bins in each dimension (z and phi)
        """
        self.r = r
        self.bin_nums = bin_nums
        self.bins = np.zeros(shape=(bin_nums, bin_nums))

    def count(self, points):
        """
        Count points and update bins.

        Parameters:
            points: np.ndarray, shape (3,) for single point or (n, 3) for multiple points
        """
        # Reshape to (n, 3) if single point
        if points.ndim == 1:
            points = points.reshape(1, 3)

        # Convert Cartesian to spherical coordinates
        x, y, z = points.T
        r = np.sqrt(x**2 + y**2 + z**2)

        # Calculate z-coordinate (normalized to [0, 1])
        z_norm = (z / r + 1) / 2

        # Calculate phi (azimuthal angle)
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi) / (2 * np.pi)

        # Map to bin indices
        z_idx = np.minimum(np.floor(z_norm * self.bin_nums), self.bin_nums - 1)
        phi_idx = np.minimum(np.floor(phi * self.bin_nums), self.bin_nums - 1)

        # Increment bin counts using np.add.at
        np.add.at(self.bins, (z_idx.astype(int), phi_idx.astype(int)), 1)

    def visualize(self, mode='normalized', show_deviation=True):
        """
        Visualize the binned data on the sphere surface.
        Creates a 2D heatmap representing the distribution in z-phi coordinates.

        Parameters:
            mode: str, visualization mode ('normalized', 'deviation', 'raw')
            show_deviation: bool, whether to show statistics about deviation
        """
        plt.figure(figsize=(10, 6))

        total_points = np.sum(self.bins)
        expected_per_bin = total_points / (self.bin_nums * self.bin_nums) if total_points > 0 else 1

        if mode == 'normalized':
            # Normalize by maximum value
            data = self.bins / (np.max(self.bins) if np.max(self.bins) > 0 else 1)
            cmap = 'viridis'
            title = 'Normalized Distribution'
            label = 'Normalized Count'
        elif mode == 'deviation':
            # Show deviation from expected uniform distribution
            data = (self.bins - expected_per_bin) / expected_per_bin
            cmap = 'coolwarm'
            title = 'Deviation from Uniform Distribution'
            label = 'Relative Deviation'
        else:  # raw
            data = self.bins
            cmap = 'viridis'
            title = 'Raw Distribution'
            label = 'Count'

        # Create a heatmap
        im = plt.imshow(data, cmap=cmap, aspect='auto', origin='lower')
        plt.colorbar(im, label=label)

        # Set labels and title
        plt.xlabel('Azimuthal Angle (φ)')
        plt.ylabel('Polar Angle (θ)')
        plt.title(title)

        # Set custom ticks for better interpretation
        plt.xticks(np.linspace(0, self.bin_nums-1, 5),
                [f'{x:.1f}π' for x in np.linspace(0, 2, 5)])
        plt.yticks(np.linspace(0, self.bin_nums-1, 5),
                [f'{x:.1f}π' for x in np.linspace(0, 1, 5)])

        if show_deviation and total_points > 0:
            std_dev = np.std(self.bins)
            cv = std_dev / expected_per_bin if expected_per_bin > 0 else 0
            plt.figtext(0.5, 0.01, f'CV: {cv:.4f}, STD: {std_dev:.2f}, Expected: {expected_per_bin:.2f}',
                        ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    def save(self, filename):
        """
        Save the bin data to a file using np.savez.

        Parameters:
            filename: str, path to save the file
        """
        np.savez(filename,
                 bins=self.bins,
                 r=self.r,
                 bin_nums=self.bin_nums)

    @classmethod
    def load(cls, filename):
        """
        Load bin data from a file.

        Parameters:
            filename: str, path to the saved file

        Returns:
            Bin: loaded Bin object
        """
        data = np.load(filename)
        bin_obj = cls(r=float(data['r']), bin_nums=int(data['bin_nums']))
        bin_obj.bins = data['bins']
        return bin_obj

def main():
    bin = Bin.load("radial_charge.npz")
    # print bin info
    print(f"bin.bin_nums: {bin.bin_nums}")
    print(f"bin.r: {bin.r}")
    bin.visualize(mode="deviation", show_deviation=True)

if __name__ == "__main__":
    main()