"""
Time Series Visualization Engine (TSVis)

Converts multivariate time series windows into publication-quality
visual representations for VLM consumption.

Design principles:
- Colorblind-safe palettes (Wong 2011)
- Consistent sizing for Qwen2-VL (448×448)
- Multi-channel line plot + optional correlation heatmap
- High DPI rendering for crisp image quality
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import seaborn as sns
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ── Colorblind-Safe Palette (Wong 2011, 7 colors) ──────────────────
CB_PALETTE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Pink
]


class TSVisualizer:
    """Render time series windows as VLM-consumable images."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (448, 448),
        dpi: int = 100,
        line_width: float = 0.5,
        palette: Optional[List[str]] = None,
        include_heatmap: bool = True,
        heatmap_ratio: float = 0.25,
        font_size: int = 8,
    ):
        self.image_size = image_size
        self.dpi = dpi
        self.line_width = line_width
        self.palette = palette or CB_PALETTE
        self.include_heatmap = include_heatmap
        self.heatmap_ratio = heatmap_ratio
        self.font_size = font_size

        # Set up matplotlib style
        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 2,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "lines.linewidth": self.line_width,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        })

    def _assign_colors(self, n_channels: int) -> List[str]:
        """Assign colors to channels, cycling through palette."""
        return [
            self.palette[i % len(self.palette)]
            for i in range(n_channels)
        ]

    def render(
        self,
        window: np.ndarray,
        labels: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Image.Image:
        """
        Render a time series window as an RGB image.

        Args:
            window: (T, D) time series window
            labels: (T,) binary anomaly labels for overlay (optional)
            channel_names: D channel/sensor names (optional)
            title: plot title (optional)

        Returns:
            PIL Image of rendered time series
        """
        T, D = window.shape
        colors = self._assign_colors(D)

        if self.include_heatmap and D >= 4:
            return self._render_with_heatmap(
                window, labels, channel_names, title, T, D, colors
            )
        else:
            return self._render_lines_only(
                window, labels, channel_names, title, T, D, colors
            )

    def _render_lines_only(
        self, window, labels, channel_names, title, T, D, colors
    ) -> Image.Image:
        """Render line chart only."""
        fig_w = self.image_size[1] / self.dpi
        fig_h = self.image_size[0] / self.dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(T)
        for d in range(D):
            label = channel_names[d] if channel_names else f"Ch{d}"
            ax.plot(x, window[:, d], color=colors[d], label=label,
                    linewidth=self.line_width, alpha=0.8)

        # Highlight anomaly regions if labels provided
        if labels is not None and labels.any():
            self._overlay_anomaly_regions(ax, labels, T, D)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title, fontweight="bold")

        # Hide legend if too many channels
        if D <= 7:
            ax.legend(fontsize=self.font_size - 2, loc="upper right",
                      ncol=min(D, 3))

        ax.grid(True, alpha=0.3, linestyle="--")

        return self._fig_to_pil(fig)

    def _render_with_heatmap(
        self, window, labels, channel_names, title, T, D, colors
    ) -> Image.Image:
        """Render line chart + correlation heatmap in vertical layout."""
        fig_w = self.image_size[1] / self.dpi
        fig_h = self.image_size[0] / self.dpi

        # Calculate subplot heights
        heatmap_h = self.heatmap_ratio
        lines_h = 1.0 - heatmap_h

        fig = plt.figure(figsize=(fig_w, fig_h))

        # Top: line plot (larger)
        gs = fig.add_gridspec(2, 1, height_ratios=[lines_h, heatmap_h],
                              hspace=0.15)
        ax_lines = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        # ── Line plot ──
        x = np.arange(T)

        # Subsample channels for visual clarity if D > 14
        if D > 14:
            step = max(1, D // 14)
            plot_indices = list(range(0, D, step))[:14]
        else:
            plot_indices = list(range(D))

        for i, d in enumerate(plot_indices):
            color = colors[d]
            label = channel_names[d] if channel_names else f"S{d}"
            ax_lines.plot(x, window[:, d], color=color, label=label,
                          linewidth=self.line_width, alpha=0.75)

        if labels is not None and labels.any():
            self._overlay_anomaly_regions(ax_lines, labels, T, D)

        if title:
            ax_lines.set_title(title, fontweight="bold", fontsize=self.font_size + 1)
        ax_lines.set_ylabel("Normalized Value", fontsize=self.font_size - 1)
        ax_lines.grid(True, alpha=0.3, linestyle="--")
        ax_lines.tick_params(labelsize=self.font_size - 2)

        # Hide x-tick labels on line plot
        ax_lines.set_xticklabels([])

        # ── Correlation heatmap ──
        corr = np.corrcoef(window.T)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(
            corr,
            ax=ax_heat,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": 0.6, "label": ""},
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
        )
        ax_heat.set_title("Correlation Matrix", fontsize=self.font_size - 1)

        return self._fig_to_pil(fig)

    def _overlay_anomaly_regions(self, ax, labels, T, y_max):
        """Highlight anomaly regions with red background."""
        in_anomaly = False
        start = 0
        for t in range(T):
            if labels[t] == 1 and not in_anomaly:
                start = t
                in_anomaly = True
            elif labels[t] == 0 and in_anomaly:
                rect = Rectangle(
                    (start, ax.get_ylim()[0]), t - start,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    facecolor="red", alpha=0.08, edgecolor=None, zorder=0
                )
                ax.add_patch(rect)
                in_anomaly = False
        if in_anomaly:
            rect = Rectangle(
                (start, ax.get_ylim()[0]), T - start,
                ax.get_ylim()[1] - ax.get_ylim()[0],
                facecolor="red", alpha=0.08, edgecolor=None, zorder=0
            )
            ax.add_patch(rect)

    def _fig_to_pil(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = BytesIO()
        fig.savefig(
            buf, format="png", dpi=self.dpi,
            bbox_inches="tight", pad_inches=0.05
        )
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize(self.image_size, Image.LANCZOS)
        return img

    def render_batch(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
    ) -> List[Image.Image]:
        """Render a batch of time series windows."""
        images = []
        for i in range(len(windows)):
            y = labels[i] if labels is not None else None
            img = self.render(
                windows[i], labels=y, channel_names=channel_names,
                title=f"Window {i}"
            )
            images.append(img)
        return images


# ── Statistical Feature Extractor ──────────────────────────────────

def extract_statistical_features(window: np.ndarray) -> np.ndarray:
    """
    Extract per-channel statistical features.

    Args:
        window: (T, D) time series window

    Returns:
        (D, 6) feature matrix: [mean, std, min, max, skewness, kurtosis]
    """
    T, D = window.shape

    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    vmin = np.min(window, axis=0)
    vmax = np.max(window, axis=0)

    # Skewness (with safe division)
    safe_std = np.where(std > 1e-8, std, 1.0)
    mu3 = np.mean((window - mean) ** 3, axis=0)
    skew = np.where(std > 1e-8, mu3 / (safe_std ** 3), 0.0)

    # Kurtosis (excess, with safe division)
    mu4 = np.mean((window - mean) ** 4, axis=0)
    kurt = np.where(std > 1e-8, mu4 / (safe_std ** 4) - 3, 0.0)

    # Final NaN guard
    skew = np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

    features = np.stack([mean, std, vmin, vmax, skew, kurt], axis=1)  # (D, 6)

    # Clip extreme values
    features = np.clip(features, -10, 10)

    return features.astype(np.float32)


def extract_enhanced_statistical_features(window: np.ndarray) -> np.ndarray:
    """
    Extract ENHANCED per-channel statistical features (12 per sensor).

    Features: mean, std, min, max, skew, kurtosis,
              rolling_mean_short, rolling_std_short,
              diff_mean, diff_std,
              spectral_centroid, spectral_bandwidth

    Args:
        window: (T, D) time series window

    Returns:
        (D, 12) enhanced feature matrix
    """
    T, D = window.shape
    basic = extract_statistical_features(window)  # (D, 6)

    # Rolling statistics (short window = T//8)
    roll_w = max(8, T // 8)
    rolling_mean = np.array([np.convolve(window[:, d], np.ones(roll_w)/roll_w, mode='valid').mean() for d in range(D)])
    rolling_std = np.array([np.convolve(window[:, d], np.ones(roll_w)/roll_w, mode='valid').std() for d in range(D)])

    # First-order difference statistics
    diff = np.diff(window, axis=0)  # (T-1, D)
    diff_mean = diff.mean(axis=0)
    diff_std = diff.std(axis=0)

    # Spectral features (via FFT)
    fft = np.abs(np.fft.rfft(window, axis=0))  # (T//2+1, D)
    freqs = np.fft.rfftfreq(T)
    spectral_centroid = np.sum(freqs[:, None] * fft, axis=0) / (np.sum(fft, axis=0) + 1e-8)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs[:, None] - spectral_centroid[None, :]) ** 2) * fft, axis=0) /
        (np.sum(fft, axis=0) + 1e-8)
    )

    # Stack enhanced features
    enhanced = np.stack([
        rolling_mean, rolling_std,
        diff_mean, diff_std,
        spectral_centroid, spectral_bandwidth,
    ], axis=1)  # (D, 6)

    # Combine basic + enhanced
    all_features = np.concatenate([basic, enhanced], axis=1)  # (D, 12)

    # Clip and NaN guard
    all_features = np.clip(np.nan_to_num(all_features, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)

    return all_features.astype(np.float32)


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic test data
    rng = np.random.default_rng(42)
    T, D = 256, 8

    # Normal data
    normal = rng.normal(0, 1, (T, D))
    # Add trend
    normal += np.linspace(0, 2, T)[:, None] * rng.normal(0, 0.3, D)

    # Anomaly at [100:130]
    anomalous = normal.copy()
    anomalous[100:130, 2] += rng.normal(5, 2, 30)
    anomalous[100:130, 5] += rng.normal(-4, 1.5, 30)

    labels = np.zeros(T, dtype=np.int8)
    labels[100:130] = 1

    viz = TSVisualizer()
    img = viz.render(anomalous, labels=labels)
    img.save("C:/Users/zengx/Desktop/CCF抽奖活动/Science Bulletin/figures/test_visualization.png")
    print(f"Test image saved. Size: {img.size}")

    features = extract_statistical_features(anomalous)
    print(f"Statistical features shape: {features.shape}")
