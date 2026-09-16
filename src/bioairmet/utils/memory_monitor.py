'''
@file    :   memory_monitor.py
@create date : 2026-01-20 09:15:20
@modify date 2026-02-19 10:25:02
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script provides tools for monitoring system resources, detecting memory leaks, and performing aggressive memory cleanup in the BioAirMet project.
    Key features include:
    - MemoryTracker: A class that records GPU and CPU memory usage over rolling windows to identify growth patterns and potential leaks from a baseline.
    - aggressive_memory_cleanup: A utility to force garbage collection across all generations and clear the CUDA cache to mitigate Out-Of-Memory (OOM) errors.
    - diagnose_system: A comprehensive diagnostic tool that logs library versions (PyTorch, h5py), hardware specs, shared memory status, and file descriptor limits.
    - BackgroundMemoryMonitor: A threaded monitor that issues warnings when GPU or CPU usage exceeds specific percentage thresholds.
    - System health functions: Helpers for logging batch-wise memory usage and verifying system stability at the start of training.
    ]
'''
import gc
import os
import time
import psutil
import threading
from typing import Optional, Dict, Any
from collections import deque

import torch


class MemoryTracker:
    """
    Tracks memory usage over time to detect leaks.
    
    Usage:
        tracker = MemoryTracker(device_id=0, window_size=100)
        for batch_idx, batch in enumerate(dataloader):
            # ... training code ...
            tracker.update(batch_idx)
            if batch_idx % 100 == 0:
                tracker.log_stats(logger)
    """
    
    def __init__(self, device_id: int = 0, window_size: int = 100):
        self.device_id = device_id
        self.window_size = window_size
        
        # Rolling windows for leak detection
        self.gpu_allocated_history: deque = deque(maxlen=window_size)
        self.gpu_reserved_history: deque = deque(maxlen=window_size)
        self.cpu_ram_history: deque = deque(maxlen=window_size)
        
        # Baseline measurements
        self.baseline_gpu_allocated = 0
        self.baseline_gpu_reserved = 0
        self.baseline_cpu_ram = 0
        self._baseline_set = False
        
    def set_baseline(self):
        """Set baseline memory levels (call after model loading)."""
        torch.cuda.synchronize(self.device_id)
        self.baseline_gpu_allocated = torch.cuda.memory_allocated(self.device_id)
        self.baseline_gpu_reserved = torch.cuda.memory_reserved(self.device_id)
        self.baseline_cpu_ram = psutil.Process().memory_info().rss
        self._baseline_set = True
        
    def update(self, batch_idx: int):
        """Record current memory usage."""
        if not self._baseline_set:
            self.set_baseline()
        
        self.gpu_allocated_history.append(
            torch.cuda.memory_allocated(self.device_id)
        )
        self.gpu_reserved_history.append(
            torch.cuda.memory_reserved(self.device_id)
        )
        self.cpu_ram_history.append(
            psutil.Process().memory_info().rss
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if len(self.gpu_allocated_history) == 0:
            return {}
        
        current_gpu_alloc = self.gpu_allocated_history[-1]
        current_gpu_reserved = self.gpu_reserved_history[-1]
        current_cpu_ram = self.cpu_ram_history[-1]
        
        # Calculate growth rates (leak detection)
        if len(self.gpu_allocated_history) >= 10:
            # Compare last 10 samples to first 10
            early_avg = sum(list(self.gpu_allocated_history)[:10]) / 10
            late_avg = sum(list(self.gpu_allocated_history)[-10:]) / 10
            gpu_growth_rate = (late_avg - early_avg) / max(early_avg, 1)
        else:
            gpu_growth_rate = 0.0
        
        return {
            'gpu_allocated_mb': current_gpu_alloc / 1024**2,
            'gpu_reserved_mb': current_gpu_reserved / 1024**2,
            'cpu_ram_mb': current_cpu_ram / 1024**2,
            'gpu_growth_from_baseline_mb': (current_gpu_alloc - self.baseline_gpu_allocated) / 1024**2,
            'cpu_growth_from_baseline_mb': (current_cpu_ram - self.baseline_cpu_ram) / 1024**2,
            'gpu_growth_rate_pct': gpu_growth_rate * 100,
            'potential_leak': gpu_growth_rate > 0.1,  # >10% growth in window
        }
    
    def log_stats(self, logger, prefix: str = ""):
        """Log memory statistics."""
        stats = self.get_stats()
        if not stats:
            return
        
        msg = (f"{prefix}GPU: {stats['gpu_allocated_mb']:.1f}MB allocated, "
               f"{stats['gpu_reserved_mb']:.1f}MB reserved | "
               f"CPU RAM: {stats['cpu_ram_mb']:.1f}MB | "
               f"Growth from baseline: GPU +{stats['gpu_growth_from_baseline_mb']:.1f}MB, "
               f"CPU +{stats['cpu_growth_from_baseline_mb']:.1f}MB")
        
        if stats['potential_leak']:
            msg += f" | ⚠️ POTENTIAL LEAK: {stats['gpu_growth_rate_pct']:.1f}% growth rate"
            logger.warning(msg)
        else:
            logger.info(msg)


def aggressive_memory_cleanup(device_id: int = 0):
    """
    Aggressive memory cleanup for end of epoch or when memory is critical.
    
    This function:
    1. Deletes all unused Python objects
    2. Runs garbage collection (multiple generations)
    3. Clears CUDA cache
    4. Synchronizes to ensure cleanup is complete
    """
    # Force garbage collection (all generations)
    gc.collect(generation=0)
    gc.collect(generation=1)
    gc.collect(generation=2)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device_id)


def get_system_memory_info() -> Dict[str, Any]:
    """Get comprehensive system memory information."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    info = {
        'cpu_total_gb': vm.total / 1024**3,
        'cpu_available_gb': vm.available / 1024**3,
        'cpu_used_pct': vm.percent,
        'swap_total_gb': swap.total / 1024**3,
        'swap_used_pct': swap.percent,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory
            
            info[f'gpu{i}_name'] = props.name
            info[f'gpu{i}_total_gb'] = total / 1024**3
            info[f'gpu{i}_allocated_gb'] = allocated / 1024**3
            info[f'gpu{i}_reserved_gb'] = reserved / 1024**3
            info[f'gpu{i}_free_gb'] = (total - reserved) / 1024**3
            info[f'gpu{i}_used_pct'] = (reserved / total) * 100
    
    return info


def diagnose_system(logger=None):
    """
    Print comprehensive system diagnostics for debugging.
    Call this at the start of training to document environment.
    """
    import h5py
    
    lines = ["=" * 60, "SYSTEM DIAGNOSTICS", "=" * 60]
    
    # Python/PyTorch versions
    lines.append(f"PyTorch: {torch.__version__}")
    lines.append(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    lines.append(f"cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    lines.append(f"h5py: {h5py.__version__}")
    lines.append(f"h5py threadsafe: {h5py.get_config().mpi}")
    
    # CPU info
    lines.append(f"\nCPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Memory info
    mem_info = get_system_memory_info()
    lines.append(f"System RAM: {mem_info['cpu_total_gb']:.1f}GB total, {mem_info['cpu_available_gb']:.1f}GB available")
    lines.append(f"Swap: {mem_info['swap_total_gb']:.1f}GB total, {mem_info['swap_used_pct']:.1f}% used")
    
    # GPU info
    if torch.cuda.is_available():
        lines.append(f"\nGPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"  GPU {i}: {mem_info[f'gpu{i}_name']}, {mem_info[f'gpu{i}_total_gb']:.1f}GB")
    else:
        lines.append("\nNo GPUs detected!")
    
    # Check shared memory (important for DataLoader)
    shm_path = '/dev/shm'
    if os.path.exists(shm_path):
        import subprocess
        try:
            result = subprocess.run(['df', '-h', shm_path], capture_output=True, text=True)
            shm_line = result.stdout.strip().split('\n')[-1].split()
            if len(shm_line) >= 4:
                lines.append(f"\n/dev/shm: {shm_line[1]} total, {shm_line[3]} available")
        except Exception:
            pass
    
    # File descriptor limits
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        lines.append(f"File Descriptor Limits: soft={soft}, hard={hard}")
        if soft < 4096:
            lines.append("  ⚠️ WARNING: Low file descriptor limit! Run: ulimit -n 65535")
    except Exception:
        pass
    
    lines.append("=" * 60)
    
    output = "\n".join(lines)
    if logger:
        for line in lines:
            logger.info(line)
    else:
        print(output)
    
    return mem_info


class BackgroundMemoryMonitor:
    """
    Background thread that monitors memory and logs warnings.
    
    Usage:
        monitor = BackgroundMemoryMonitor(device_id=0, warning_threshold_pct=85)
        monitor.start()
        # ... training ...
        monitor.stop()
    """
    
    def __init__(self, device_id: int = 0, interval_seconds: float = 30.0,
                 warning_threshold_pct: float = 85.0, logger=None):
        self.device_id = device_id
        self.interval = interval_seconds
        self.threshold = warning_threshold_pct
        self.logger = logger
        self.running = False
        self._thread: Optional[threading.Thread] = None
    
    def _monitor_loop(self):
        while self.running:
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(self.device_id)
                    reserved = torch.cuda.memory_reserved(self.device_id)
                    used_pct = (reserved / props.total_memory) * 100
                    
                    if used_pct > self.threshold:
                        msg = (f"⚠️ GPU {self.device_id} memory critical: "
                               f"{used_pct:.1f}% used ({reserved/1024**3:.1f}GB)")
                        if self.logger:
                            self.logger.warning(msg)
                        else:
                            print(msg)
                
                # Also check CPU RAM
                vm = psutil.virtual_memory()
                if vm.percent > self.threshold:
                    msg = f"⚠️ CPU RAM critical: {vm.percent:.1f}% used"
                    if self.logger:
                        self.logger.warning(msg)
                    else:
                        print(msg)
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Memory monitor error: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start background monitoring."""
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)


def log_memory_every_n_batches(batch_idx: int, n: int, device_id: int, logger, prefix: str = ""):
    """
    Convenience function to log memory every N batches.
    
    Usage:
        for batch_idx, batch in enumerate(dataloader):
            # ... training ...
            log_memory_every_n_batches(batch_idx, 100, device_id, logger)
    """
    if batch_idx % n == 0:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            logger.info(f"{prefix}Batch {batch_idx} | GPU Memory: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")


def check_for_memory_leaks(tracker: MemoryTracker, logger, critical_threshold_mb: float = 500.0) -> bool:
    """
    Check if there's a memory leak based on tracker history.
    
    Returns:
        bool: True if a potential leak is detected
    """
    stats = tracker.get_stats()
    if not stats:
        return False
    
    is_leaking = stats['potential_leak']
    if is_leaking:
        logger.warning(
            f"⚠️ MEMORY LEAK DETECTED: GPU memory growing at {stats['gpu_growth_rate_pct']:.1f}% rate. "
            f"Growth since baseline: {stats['gpu_growth_from_baseline_mb']:.1f}MB"
        )
    
    # Also check absolute growth
    if stats['gpu_growth_from_baseline_mb'] > critical_threshold_mb:
        logger.warning(
            f"⚠️ CRITICAL: GPU memory grew {stats['gpu_growth_from_baseline_mb']:.1f}MB from baseline!"
        )
        is_leaking = True
    
    return is_leaking
