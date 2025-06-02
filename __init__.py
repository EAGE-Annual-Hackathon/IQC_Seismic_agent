from server import IQCAnalyzer, analyzer as _analyzer, mcp
from fastmcp import Image
from typing import Optional, Dict, Any

analyzer: IQCAnalyzer = _analyzer

def load_file(filename: str) -> Dict[str, Any]:
    """Load a SEG-Y volume for QC analysis."""
    return analyzer.read_segy_file(filename)

def get_statistics() -> Dict[str, Any]:
    """Retrieve statistics of the currently loaded SEG-Y data."""
    return analyzer.calculate_statistics()

def plot_inline_section(index: Optional[int] = None) -> Image:
    """Plot an inline section at the specified inline index."""
    return Image(path=analyzer._plot_inline_section(index))

def plot_amplitude_histogram() -> Image:
    """Generate and return the amplitude histogram for the loaded data."""
    return Image(path=analyzer._plot_amplitude_histogram())

def plot_frequency_spectrum() -> Image:
    """Generate and return the average frequency spectrum plot."""
    return Image(path=analyzer._plot_frequency_spectrum())

def plot_qc_summary() -> Image:
    """Generate and return a summary bar chart of QC indicators."""
    return Image(path=analyzer._plot_qc_summary())

def detect_bad_traces() -> Dict[str, Any]:
    """Identify traces with abnormal RMS amplitudes."""
    return analyzer.detect_bad_traces()

def adjust_thresholds(low_factor: float, high_factor: float) -> Dict[str, Any]:
    """Adjust low/high RMS thresholds for bad-trace detection."""
    return analyzer.adjust_thresholds(low_factor, high_factor)

def get_identity() -> Dict[str, str]:
    """Return the IQC Assistant identity and role description."""
    return {
        "name": "IQC Assistant",
        "description": "An intelligent agent for seismic quality control and optimization.",
        "role": analyzer  # role text is stored in PREPROMPT_TEXT inside server
    }

