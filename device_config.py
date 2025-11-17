"""
Device Configuration for PyTorch
=================================

Supports CUDA (NVIDIA), MPS (Apple Metal), and CPU backends.
Automatically detects and selects the best available device.
"""

import torch
from typing import Optional, Literal

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


class DeviceConfig:
    """Manages device selection for PyTorch models."""
    
    def __init__(self, device: DeviceType = "auto", verbose: bool = True):
        """
        Initialize device configuration.
        
        Args:
            device: Device type - "cuda", "mps", "cpu", or "auto"
            verbose: Print device information
        """
        self.requested_device = device
        self.device = self._select_device(device)
        self.device_type = self._get_device_type()
        
        if verbose:
            self._print_device_info()
    
    def _select_device(self, device: DeviceType) -> torch.device:
        """Select the appropriate device."""
        if device == "auto":
            return self._auto_select_device()
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("⚠️  CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
        elif device == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                print("⚠️  MPS (Apple Metal) requested but not available. Falling back to CPU.")
                return torch.device("cpu")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device type: {device}")
    
    def _auto_select_device(self) -> torch.device:
        """Automatically select the best available device."""
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _get_device_type(self) -> str:
        """Get device type as string."""
        return str(self.device).split(":")[0]
    
    def _print_device_info(self):
        """Print information about selected device."""
        print("\n" + "=" * 60)
        print("🖥️  DEVICE CONFIGURATION")
        print("=" * 60)
        
        if self.device_type == "cuda":
            print(f"✓ Using NVIDIA CUDA")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.device_type == "mps":
            print(f"✓ Using Apple Metal (MPS)")
            print(f"  Device: Apple Silicon GPU")
            # MPS doesn't provide memory info easily
        else:
            print(f"✓ Using CPU")
            print(f"  Note: Training will be slower than GPU")
        
        print("=" * 60 + "\n")
    
    def get_device_str(self) -> str:
        """Get device as string for logging."""
        if self.device_type == "cuda":
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return self.device_type
    
    @staticmethod
    def get_available_devices() -> dict:
        """Get information about all available devices."""
        devices = {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
        }
        
        info = {"available": devices}
        
        if devices["cuda"]:
            info["cuda_details"] = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            }
        
        if devices["mps"]:
            info["mps_details"] = {
                "backend": "Apple Metal Performance Shaders",
                "available": True,
            }
        
        return info
    
    @staticmethod
    def print_all_available():
        """Print all available devices."""
        print("\n" + "=" * 60)
        print("AVAILABLE DEVICES")
        print("=" * 60)
        
        info = DeviceConfig.get_available_devices()
        
        print(f"\n✓ CPU: Always available")
        
        if info["available"]["cuda"]:
            print(f"\n✓ CUDA (NVIDIA GPU):")
            details = info["cuda_details"]
            print(f"  - Devices: {details['device_count']}")
            print(f"  - Name: {details['device_name']}")
            print(f"  - CUDA Version: {details['cuda_version']}")
        else:
            print(f"\n✗ CUDA: Not available")
        
        if info["available"]["mps"]:
            print(f"\n✓ MPS (Apple Metal):")
            print(f"  - Backend: Apple Metal Performance Shaders")
            print(f"  - Available for M1/M2/M3 Macs")
        else:
            print(f"\n✗ MPS: Not available (requires Apple Silicon Mac)")
        
        print("\n" + "=" * 60 + "\n")
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to configured device."""
        return tensor.to(self.device)
    
    def empty_cache(self):
        """Clear device cache (for CUDA/MPS)."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()


def get_optimal_device(prefer: Optional[DeviceType] = None, verbose: bool = True) -> str:
    """
    Get optimal device string for PyTorch.
    
    Args:
        prefer: Preferred device type ("cuda", "mps", "cpu", or None for auto)
        verbose: Print device selection info
    
    Returns:
        Device string (e.g., "cuda", "mps", "cpu")
    
    Examples:
        >>> device = get_optimal_device()  # Auto-select
        >>> device = get_optimal_device(prefer="mps")  # Prefer Apple Metal
        >>> device = get_optimal_device(prefer="cuda")  # Prefer NVIDIA
    """
    config = DeviceConfig(device=prefer or "auto", verbose=verbose)
    return config.get_device_str()


def check_device_availability():
    """Print summary of available devices."""
    DeviceConfig.print_all_available()


if __name__ == "__main__":
    # Print all available devices
    check_device_availability()
    
    # Test auto-selection
    print("\nTesting auto device selection:")
    config = DeviceConfig(device="auto", verbose=True)
    
    # Test tensor operation
    print("\nTesting tensor operation:")
    x = torch.randn(100, 100)
    x = config.to_device(x)
    y = x @ x.T
    print(f"✓ Successfully computed on {config.device_type}")
