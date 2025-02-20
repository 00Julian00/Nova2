"""
Holds various functionality for memory management.
"""

import os
import math
from pathlib import Path

from transformers import AutoModel
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
import psutil
import nvidia_smi

class MemCheckResult:
    def __init__(
                self,
                ram: int,
                vram: int,
                required_ram: int,
                required_vram: int
                ) -> None:
        self.total_ram = ram
        self.total_vram = vram
        self.required_ram = required_ram
        self.required_vram = required_vram

class MemoryManager:
    def __init__(self) -> None:
        pass
    
    @DeprecationWarning("This function should not be called. It is broken.")
    def estimate_memory_requirement(self, model: str, precision: str = "fp16") -> float:
        """
        Calculates a rough estimate of how much memory a model will require. Result is in MB.
        """
        api = HfApi()

        safetensor_files = []

        try:
            files = api.list_repo_files(repo_id=model)

            for file in files:
                if file.endswith(".safetensors"):
                    safetensor_files.append(file)
        except Exception as e:
            print(f"Error: {e}")

        base_size = 0
        for file in safetensor_files:
            url = f"https://huggingface.co/{model}/blob/main/{file}"
            try:
                metadata = get_hf_file_metadata(url=url)
                base_size += metadata.size
            except Exception as e:
                print(f"Error getting metadata for {file}: {e}")

        precision_multiplier = 2 if precision == "fp16" else 4
        base_memory = base_size * (precision_multiplier / 4)

        activation_multiplier = 2.5
        estimated_vram = base_memory * activation_multiplier

        return math.ceil(estimated_vram / (1024 * 1024))

    def construct_mem_check_result(self, ram_required: int, vram_required: int) -> MemCheckResult:
        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        total_ram = math.ceil(psutil.virtual_memory().total / (1024 * 1024))

        total_vram = math.ceil(gpu_info.total / (1024 * 1024))

        nvidia_smi.nvmlShutdown()

        return MemCheckResult(ram=total_ram, vram=total_vram, required_ram=ram_required, required_vram=vram_required)

    def _get_safetensors_path(self, model: str) -> list[str]:
        model = AutoModel.from_pretrained(model)
    
        # Get cache directory where model is stored
        cache_dir = model.config.cache_dir
        if cache_dir is None:
            # Default HF cache location
            cache_dir = os.path.join(Path.home(), '.cache', 'huggingface', 'hub')
        
        # Look for .safetensors files
        model_dir = os.path.join(cache_dir, model)
        safetensor_files = list(Path(model_dir).glob("*.safetensors"))
        
        return safetensor_files
