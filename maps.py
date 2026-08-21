import matplotlib.pyplot as plt
import colorsys
import numpy as np

from pose import Pose

class Landmark:
    def __init__(self, pos: Pose, signature: any):
        self.pos = pos
        self.signature = signature

    def draw(self, ax: plt.Axes, color: str = 'red', set_label = False):
        ax.scatter(self.pos.x, self.pos.y, c=color, label = f"Landmark {self.signature}" if set_label else None)

class Map: pass

class LandmarkMap(Map):
    def __init__(self, landmarks: list[Landmark] | None = None, x_lims: tuple[float, float] = (-5, 5), y_lims: tuple[float, float] = (-5, 5)):
        super().__init__()

        if landmarks is None:
            landmarks = []

        self.landmarks = landmarks
        self.colors = []
        self.x_lims = x_lims
        self.y_lims = y_lims        

        self._generateLandmarkColors()

    def _generateLandmarkColors(self):
        n = len(self.landmarks)

        if n == 0: return

        self.colors = []

        for i in range(n):
            
            hue = (i + 1) / n
            saturation = (90 + np.random.rand() * 10) / 100
            value = (50 + np.random.rand() * 10) / 100 # lightness

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)

            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

            self.colors.append(hex_color)


    def draw(self, ax: plt.Axes, landmark_labels: bool = False):
        for l_idx, l in enumerate(self.landmarks):
            l.draw(ax, color = self.colors[l_idx], set_label=landmark_labels)

        ax.set_xlim(self.x_lims[0], self.x_lims[1])
        ax.set_ylim(self.y_lims[0], self.y_lims[1])

