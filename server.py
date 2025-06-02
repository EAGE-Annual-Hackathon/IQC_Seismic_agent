#!/usr/bin/env python3
"""
IQC Assistant : Seismic data Quality Control MCP Agent server
An intelligent agent designed for interactive quality control and optimization of seismic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import segyio
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.signal import butter, filtfilt
from fastmcp import FastMCP, Image

from config import BASE_PATH, DATA_PATH, FIGURES_PATH, PREPROMPT_PATH, MCP_SERVER_URL

if PREPROMPT_PATH.exists():
    with open(PREPROMPT_PATH, 'r', encoding='utf-8') as f:
        PREPROMPT_TEXT = f.read()
else:
    print('no preprompt file found, not using preprompt')
    PREPROMPT_TEXT = ''

mcp = FastMCP("IQC Assistant")

class IQCAnalyzer:
    """Class to handle 2D SEG-Y quality control operations"""

    def __init__(self):
        self.current_data = None  # shape: (samples, inlines, xlines)
        self.current_samples = None
        self.file_info = {}
        self.fig_dir = FIGURES_PATH
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.low_rms_factor = 0.1
        self.high_rms_factor = 3.0

    def read_segy_file(self, filename: str) -> Dict[str, Any]:
        path = DATA_PATH / filename
        if not path.exists():
            return {"status": "error", "message": f"File not found: {filename}"}
        try:
            with segyio.open(str(path), ignore_geometry=False) as f:
                f.mmap()
                inlines = f.ilines
                xlines = f.xlines
                cube = segyio.tools.cube(f)  # (inlines, xlines, samples)
                self.current_data = np.transpose(cube, (2, 0, 1))
                self.current_samples = f.samples
                self.file_info = {
                    "filename": filename,
                    "samples_per_trace": len(f.samples),
                    "sample_interval_ms": segyio.dt(f) / 1000,
                    "inlines": len(inlines),
                    "crosslines": len(xlines),
                }
                return {
                    "status": "success",
                    "file_info": self.file_info,
                    "message": f"Loaded SEG-Y: {len(inlines)} inlines × {len(xlines)} crosslines",
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def calculate_statistics(self) -> Dict[str, Any]:
        if self.current_data is None:
            return {"status": "error", "message": "No data loaded"}
        s, ninl, nxln = self.current_data.shape
        flat = self.current_data.reshape(s, ninl * nxln)
        rms = np.sqrt(np.mean(flat**2, axis=0))
        median_rms = float(np.median(rms))
        return {
            "status": "success",
            "statistics": {
                "data_range": {"min": float(np.min(flat)), "max": float(np.max(flat))},
                "mean_amplitude": float(np.mean(flat)),
                "std_amplitude": float(np.std(flat)),
                "zero_traces": int(np.sum(np.all(flat == 0, axis=0))),
                "nan_values": int(np.sum(np.isnan(flat))),
                "rms_amplitudes": rms.tolist(),
                "quality_indicators": {
                    "low_amplitude_traces": int(np.sum(rms < self.low_rms_factor * median_rms)),
                    "high_amplitude_traces": int(np.sum(rms > self.high_rms_factor * median_rms)),
                    "median_rms": median_rms,
                },
            },
        }

    def apply_bandpass_filter(self, low_hz: float, high_hz: float, order: int = 4) -> Dict[str, Any]:
        if self.current_data is None:
            return {"status": "error", "message": "No data loaded"}
        fs = 1000.0 / self.file_info.get("sample_interval_ms", 4)
        nyq = 0.5 * fs
        low = low_hz / nyq
        high = high_hz / nyq
        if not (0 < low < high < 1.0):
            return {"status": "error", "message": f"Invalid band: {low_hz}-{high_hz} Hz"}
        b, a = butter(order, [low, high], btype="band")
        s, ninl, nxln = self.current_data.shape
        reshaped = self.current_data.reshape(s, ninl * nxln)
        try:
            filtered = filtfilt(b, a, reshaped, axis=0)
        except Exception as e:
            return {"status": "error", "message": str(e)}
        self.current_data = filtered.reshape(s, ninl, nxln)
        return {"status": "success", "message": f"Applied {low_hz}-{high_hz} Hz bandpass"}

    def _save_plot(self, fig: plt.Figure, name: str) -> str:
        out = self.fig_dir / f"{name}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out)

    def _plot_inline(self, idx: Optional[int] = None) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        ninl = self.current_data.shape[1]
        if idx is None or not (0 <= idx < ninl):
            idx = ninl // 2
        sec = self.current_data[:, idx, :]
        fig, ax = plt.subplots(figsize=(12, 8))
        extent = [0, sec.shape[1], self.current_samples[-1], self.current_samples[0]]
        im = ax.imshow(sec, aspect="auto", cmap="seismic",
                       extent=extent,
                       vmin=-np.percentile(np.abs(sec), 95),
                       vmax=np.percentile(np.abs(sec), 95))
        ax.set_xlabel("Crossline"); ax.set_ylabel("Time (ms)")
        ax.set_title(f"Inline {idx}")
        plt.colorbar(im, ax=ax)
        return self._save_plot(fig, f"inline_{idx}")

    def _plot_crossline(self, idx: Optional[int] = None) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        nxln = self.current_data.shape[2]
        if idx is None or not (0 <= idx < nxln):
            idx = nxln // 2
        sec = self.current_data[:, :, idx]
        fig, ax = plt.subplots(figsize=(12, 8))
        extent = [0, sec.shape[1], self.current_samples[-1], self.current_samples[0]]
        im = ax.imshow(sec, aspect="auto", cmap="seismic",
                       extent=extent,
                       vmin=-np.percentile(np.abs(sec), 95),
                       vmax=np.percentile(np.abs(sec), 95))
        ax.set_xlabel("Inline"); ax.set_ylabel("Time (ms)")
        ax.set_title(f"Crossline {idx}")
        plt.colorbar(im, ax=ax)
        return self._save_plot(fig, f"crossline_{idx}")

    def _plot_histogram(self) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.current_data.flatten(), bins=100, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Amplitude"); ax.set_ylabel("Frequency")
        ax.set_title("Amplitude Histogram"); ax.grid(True, alpha=0.3)
        return self._save_plot(fig, "histogram")

    def _plot_spectrum(self) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        s, ninl, nxln = self.current_data.shape
        flat = self.current_data.reshape(s, ninl * nxln)
        spec = np.abs(np.fft.fft(flat, axis=0))
        avg = np.mean(spec, axis=1)
        dt = self.file_info.get("sample_interval_ms", 4) / 1000
        freqs = np.fft.fftfreq(len(self.current_samples), dt)[: len(self.current_samples) // 2]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, avg[: len(freqs)], linewidth=1)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude")
        ax.set_title("Average Spectrum"); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(250, freqs[-1]))
        return self._save_plot(fig, "spectrum")

    def _plot_qc_summary(self) -> str:
        stats = self.calculate_statistics()["statistics"]
        labels = ["Zero Traces", "NaN Values", "Low Amp", "High Amp"]
        vals = [
            stats["zero_traces"],
            stats["nan_values"],
            stats["quality_indicators"]["low_amplitude_traces"],
            stats["quality_indicators"]["high_amplitude_traces"],
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, vals, color=["red", "orange", "blue", "purple"], alpha=0.7)
        ax.set_ylabel("Count"); ax.set_title("QC Summary"); ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center")
        return self._save_plot(fig, "qc_summary")

    def _plot_sample_traces(self) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        s, ninl, nxln = self.current_data.shape
        indices = [0, ninl * nxln // 4, ninl * nxln // 2, 3 * ninl * nxln // 4, ninl * nxln - 1]
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in indices:
            il = i // nxln; xl = i % nxln
            tr = self.current_data[:, il, xl]
            ax.plot(tr, self.current_samples, label=f"IL{il}-XL{xl}")
        ax.set_xlabel("Amplitude"); ax.set_ylabel("Time (ms)")
        ax.set_title("Sample Traces"); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        return self._save_plot(fig, "sample_traces")

    def detect_bad_traces(self) -> Dict[str, Any]:
        if self.current_data is None:
            return {"status": "error", "message": "No data loaded"}
        s, ninl, nxln = self.current_data.shape
        flat = self.current_data.reshape(s, ninl * nxln)
        rms = np.sqrt(np.mean(flat**2, axis=0))
        med = float(np.median(rms))
        low_t = self.low_rms_factor * med
        high_t = self.high_rms_factor * med
        return {
            "status": "success",
            "low_indices": np.where(rms < low_t)[0].tolist(),
            "high_indices": np.where(rms > high_t)[0].tolist(),
            "low_threshold": low_t,
            "high_threshold": high_t,
            "median_rms": med,
        }

    def adjust_thresholds(self, low: float, high: float) -> Dict[str, Any]:
        self.low_rms_factor = low
        self.high_rms_factor = high
        return {"status": "success", "low_rms_factor": low, "high_rms_factor": high}

    def get_trace_statistics(self, trace_indices: List[int]) -> Dict[str, Any]:
        if self.current_data is None:
            return {"status": "error", "message": "No data loaded"}
        s, ninl, nxln = self.current_data.shape
        total = ninl * nxln
        valid = [i for i in trace_indices if 0 <= i < total]
        if not valid:
            return {"status": "error", "message": "No valid indices"}
        stats: Dict[str, Any] = {}
        for i in valid:
            il, xl = divmod(i, nxln)
            tr = self.current_data[:, il, xl]
            stats[f"trace_{i}"] = {
                "min": float(np.min(tr)),
                "max": float(np.max(tr)),
                "mean": float(np.mean(tr)),
                "std": float(np.std(tr)),
                "rms": float(np.sqrt(np.mean(tr**2))),
                "zero_samples": int(np.sum(tr == 0)),
                "nan_samples": int(np.sum(np.isnan(tr))),
            }
        return {"status": "success", "trace_statistics": stats}

    def list_files(self) -> Dict[str, Any]:
        directory = DATA_PATH
        d = Path(directory) if directory else Path.cwd()
        if not d.exists() or not d.is_dir():
            return {"status": "error", "message": f"Invalid directory: {d}"}
        return {"status": "success", "files": sorted([p.name for p in d.iterdir() if p.is_file()])}

    def list_available_tools(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "available_tools": {
                "load_segy_file": "Load a SEG-Y file",
                "get_file_info": "Get basic stats and QC metrics",
                "apply_bandpass_filter": "Apply Butterworth bandpass",
                "show_inline_section": "Plot inline section",
                "show_crossline_section": "Plot crossline section",
                "show_amplitude_histogram": "Plot amplitude histogram",
                "show_frequency_spectrum": "Plot average spectrum",
                "show_qc_summary": "Plot QC summary",
                "show_sample_traces": "Plot sample traces",
                "show_rms_map": "Plot 2D RMS amplitude map",
                "detect_bad_traces": "Identify abnormal RMS traces",
                "adjust_qc_thresholds": "Adjust RMS thresholds",
                "get_trace_statistics": "Get stats for trace indices",
                "list_files": "List dataset files",
                "list_available_tools": "List all tools",
                "iqc_identity": "Return identity and role",
            },
        }

    def _plot_rms_map(self) -> str:
        if self.current_data is None:
            raise RuntimeError("No data loaded")
        s, ninl, nxln = self.current_data.shape
        flat = self.current_data.reshape(s, ninl * nxln)
        rms = np.sqrt(np.mean(flat**2, axis=0))
        rms2d = rms.reshape(ninl, nxln)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(rms2d, aspect="auto", cmap="viridis")
        ax.set_xlabel("Crossline"); ax.set_ylabel("Inline")
        ax.set_title("RMS Amplitude Map")
        plt.colorbar(im, ax=ax)
        return self._save_plot(fig, "rms_map")


analyzer = IQCAnalyzer()


@mcp.tool()
def load_segy_file(filename: str) -> Dict[str, Any]:
    return analyzer.read_segy_file(filename)


@mcp.tool()
def get_file_info() -> Dict[str, Any]:
    return analyzer.calculate_statistics()


@mcp.tool()
def apply_bandpass_filter(low_hz: float, high_hz: float, order: int = 4) -> Dict[str, Any]:
    return analyzer.apply_bandpass_filter(low_hz, high_hz, order)


@mcp.tool()
def show_inline_section(index: Optional[int] = None) -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_inline(index))


@mcp.tool()
def show_crossline_section(index: Optional[int] = None) -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_crossline(index))


@mcp.tool()
def show_amplitude_histogram() -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_histogram())


@mcp.tool()
def show_frequency_spectrum() -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_spectrum())


@mcp.tool()
def show_qc_summary() -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_qc_summary())


@mcp.tool()
def show_sample_traces() -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_sample_traces())


@mcp.tool()
def show_rms_map() -> Image:
    if analyzer.current_data is None:
        return Image.blank("No data loaded")
    return Image(path=analyzer._plot_rms_map())


@mcp.tool()
def detect_bad_traces() -> Dict[str, Any]:
    return analyzer.detect_bad_traces()


@mcp.tool()
def adjust_qc_thresholds(low: float, high: float) -> Dict[str, Any]:
    return analyzer.adjust_thresholds(low, high)


@mcp.tool()
def get_trace_statistics(trace_indices: List[int]) -> Dict[str, Any]:
    return analyzer.get_trace_statistics(trace_indices)


@mcp.tool()
def list_files() -> Dict[str, Any]:
    return analyzer.list_files()


@mcp.tool()
def list_available_tools() -> Dict[str, Any]:
    return analyzer.list_available_tools()


@mcp.tool()
def iqc_identity() -> Dict[str, str]:
    return {
        "name": "IQC Assistant",
        "description": "An intelligent agent for seismic quality control and optimization.",
        "role": PREPROMPT_TEXT,
    }

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=9001,
        path="/IQC",
        log_level="debug"
    )
