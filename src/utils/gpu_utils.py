# src/utils/gpu_utils.py
"""
GPU Detection and Device Management Utilities.

Automatically detects CUDA GPUs and configures optimal device settings
for LLM inference and embedding computation.
"""

import os
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None


@dataclass
class DeviceConfig:
    """Configuration for compute device."""
    device: str  # "cuda", "cuda:0", "cuda:1", "mps", "cpu"
    device_name: str  # Human-readable name
    is_gpu: bool
    memory_mb: Optional[int] = None

    def __str__(self) -> str:
        if self.is_gpu:
            mem = f" ({self.memory_mb}MB)" if self.memory_mb else ""
            return f"{self.device_name}{mem}"
        return self.device_name


def detect_cuda_gpus() -> List[GPUInfo]:
    """
    Detect available CUDA GPUs using nvidia-smi.

    Returns:
        List of GPUInfo objects for each detected GPU.
    """
    gpus = []

    try:
        # Query nvidia-smi for GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append(GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total_mb=int(float(parts[2])),
                            memory_free_mb=int(float(parts[3])),
                            driver_version=parts[4]
                        ))

        # Get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Try to get CUDA version from nvcc
        try:
            nvcc_result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvcc_result.returncode == 0:
                for line in nvcc_result.stdout.split("\n"):
                    if "release" in line.lower():
                        # Extract version like "release 12.1"
                        parts = line.split("release")
                        if len(parts) > 1:
                            cuda_ver = parts[1].split(",")[0].strip()
                            for gpu in gpus:
                                gpu.cuda_version = cuda_ver
        except FileNotFoundError:
            pass

    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPU available")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return gpus


def detect_mps() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def detect_torch_cuda() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def get_optimal_device() -> DeviceConfig:
    """
    Automatically detect and return the optimal compute device.

    Priority:
    1. CUDA GPU (if available)
    2. Apple MPS (if available)
    3. CPU (fallback)

    Returns:
        DeviceConfig with optimal device settings.
    """
    # Check CUDA GPUs
    gpus = detect_cuda_gpus()
    if gpus:
        # Use GPU with most free memory
        best_gpu = max(gpus, key=lambda g: g.memory_free_mb)
        return DeviceConfig(
            device=f"cuda:{best_gpu.index}" if len(gpus) > 1 else "cuda",
            device_name=best_gpu.name,
            is_gpu=True,
            memory_mb=best_gpu.memory_free_mb
        )

    # Check PyTorch CUDA (backup method)
    torch_cuda, cuda_ver = detect_torch_cuda()
    if torch_cuda:
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return DeviceConfig(
                device="cuda",
                device_name=name,
                is_gpu=True,
                memory_mb=mem
            )
        except Exception:
            return DeviceConfig(
                device="cuda",
                device_name="CUDA GPU",
                is_gpu=True
            )

    # Check Apple MPS
    if detect_mps():
        return DeviceConfig(
            device="mps",
            device_name="Apple Metal (MPS)",
            is_gpu=True
        )

    # Fallback to CPU
    return DeviceConfig(
        device="cpu",
        device_name="CPU",
        is_gpu=False
    )


def check_ollama_gpu() -> Tuple[bool, str]:
    """
    Check if Ollama is using GPU acceleration.

    Ollama automatically uses GPU if available. This function
    checks the Ollama process to verify GPU usage.

    Returns:
        Tuple of (is_using_gpu, status_message)
    """
    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama not running"

        # Check nvidia-smi for Ollama GPU usage
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout.lower()
            if "ollama" in output:
                return True, "Ollama is using GPU"
            else:
                # Ollama may not show up if no inference is running
                # Check if GPU is available
                gpus = detect_cuda_gpus()
                if gpus:
                    return True, f"GPU available: {gpus[0].name} (Ollama will use GPU)"
                return False, "No GPU detected"

        return False, "Could not verify GPU usage"

    except Exception as e:
        return False, f"Check failed: {e}"


def print_gpu_status():
    """Print GPU detection status to console."""
    print("\n" + "=" * 60)
    print("GPU STATUS")
    print("=" * 60)

    # Detect GPUs
    gpus = detect_cuda_gpus()

    if gpus:
        print(f"\n✓ Found {len(gpus)} NVIDIA GPU(s):\n")
        for gpu in gpus:
            print(f"  [{gpu.index}] {gpu.name}")
            print(f"      Memory: {gpu.memory_free_mb:,} MB free / {gpu.memory_total_mb:,} MB total")
            if gpu.cuda_version:
                print(f"      CUDA: {gpu.cuda_version}")
            if gpu.driver_version:
                print(f"      Driver: {gpu.driver_version}")
            print()
    else:
        # Check MPS
        if detect_mps():
            print("\n✓ Apple Metal (MPS) available\n")
        else:
            print("\n✗ No GPU detected - will use CPU\n")

    # Check Ollama GPU status
    ollama_gpu, ollama_msg = check_ollama_gpu()
    print(f"Ollama: {ollama_msg}")

    # PyTorch status
    torch_cuda, cuda_ver = detect_torch_cuda()
    if torch_cuda:
        print(f"PyTorch CUDA: {cuda_ver}")

    # Optimal device
    device = get_optimal_device()
    print(f"\nOptimal device: {device}")

    print("=" * 60 + "\n")

    return device


def configure_environment_for_gpu():
    """
    Configure environment variables for optimal GPU usage.

    Sets:
    - CUDA_VISIBLE_DEVICES (if multiple GPUs)
    - PYTORCH_CUDA_ALLOC_CONF (memory optimization)
    """
    gpus = detect_cuda_gpus()

    if gpus:
        # Use GPU with most free memory
        best_gpu = max(gpus, key=lambda g: g.memory_free_mb)

        # Set CUDA device
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu.index)
            logger.info(f"Set CUDA_VISIBLE_DEVICES={best_gpu.index}")

        # Memory optimization for PyTorch
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        return best_gpu

    return None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = print_gpu_status()
