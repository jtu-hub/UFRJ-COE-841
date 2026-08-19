# %%
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from angle import Angle

class ProbDistribution:
    def __init__(self, variance, mean = 0):
        self.mean = mean
        self.variance = variance

    def sample(self):
        raise NotImplemented
    
    def prob(self, error):
        raise NotImplemented

class GaussianDistribution(ProbDistribution):
    def __init__(self, variance, mu = 0):
        super().__init__(variance, mean = mu)

        self.mu = mu
        self.sigma = np.sqrt(variance)

    def sample(self):
        samples = np.random.uniform(-self.sigma, self.sigma, 12)

        return np.sum(samples) / 2 

    def prob(self, error):
        if self.sigma <= 0:
            return 0.0
        
        coefficient = 1.0 / (np.sqrt(2 * np.pi) * self.sigma)
        exponent = -0.5 * (error ** 2) / (self.variance)

        return coefficient * np.exp(exponent)

    @staticmethod
    def plotGaussian2D(ax: plt.Axes, mu: np.array, sigma: np.array, n_sigmas: int = 3, color: str | None = None, label: str | None = None, draw_all_sigma: bool = True, draw_mu: bool = True, add_legend: bool = True,  **kwargs):
        if mu.shape != (2, 1):
            raise ValueError(f"Wrong dimentison for mu, should be (2, 1) but"
                            f"an array of shape {mu.shape} was provided")

        if sigma.shape != (2, 2):
            raise ValueError(f"Wrong dimentison for sigma, should be (2, 2) but"
                            f" an array of shape {sigma.shape} was provided")

        mu_x = mu[0][0]
        mu_y = mu[1][0]

        s_xx = sigma[0][0]
        s_yy = sigma[1][1]

        s_xy = sigma[0][1]
        s_yx = sigma[1][0]

        th = Angle(0.)
        
        if not np.isclose(s_xy, s_yx, rtol=1e-9, atol=1e-12):
            raise ValueError(f"Assymetrical Standard Deviation Matrix\nSigma: {s_xx:.2f} {s_xy:.2f}\n       {s_yx:.2f} {s_yy:.2f}")

        if s_xy != 0:
            #eigen value decomposition to find the dierction of the gaussian
            eig_val, eig_vec = np.linalg.eig(sigma)

            th = Angle.atan2(eig_vec[1][0], eig_vec[0][0])

            s_xx = eig_val[0]
            s_yy = eig_val[1]

        s_label = r"$\sigma$"
        m_label = r"$\mu$"

        if label is not None:
            s_label += f" {label}"
            m_label += f" {label}"

        for n in range(n_sigmas):
            if draw_all_sigma or (not draw_all_sigma and n == n_sigmas - 1):
                ax.add_patch(
                    Ellipse(
                        (mu_x, mu_y),
                        (n + 1) * 2 * np.sqrt(s_xx),
                        (n + 1) * 2 * np.sqrt(s_yy),
                        angle=th.deg,
                        color=f"#{'%02x' % int(255 / n_sigmas * (n_sigmas - n))}0000" if color is None else color,
                        fill=False,
                        linestyle='--',
                        label=f"{n + 1}" + s_label if add_legend else None
                    )
                )

        if draw_mu:
            ax.scatter(mu_x, mu_y, color="#ff0000" if color is None else color, label=m_label if add_legend else None)

        ax.legend()

class TriangularDistribution(ProbDistribution):
    def __init__(self, variance):
            super().__init__(variance, mean = 0)
    
    def sample(self):
        samples = np.random.uniform(-np.sqrt(self.variance), np.sqrt(self.variance), 2)

        return np.sqrt(6) * np.sum(samples) / 2 

    def prob(self, error):
        if self.sigma <= 0:
            return 0.0
        b_sq = 6 * self.variance
        b = 1 / np.sqrt(b_sq)
        c = np.abs(error) / b_sq

        return np.max(b - c, 0)
# %%


