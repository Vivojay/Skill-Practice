"""Small shared logging/checkpoint helpers for the four concrete experiments."""
import ctypes
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import torch


def setup(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)


def metadata(config):
    return dict(config=config, seed=config.get('seed'), python=platform.python_version(),
                torch=str(torch.__version__), device='cpu', dtype='float32',
                platform=platform.platform(), threads=torch.get_num_threads(),
                command=subprocess.list2cmdline(sys.orig_argv),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def timing(values):
    values=sorted(values)
    return dict(median_ms=statistics.median(values)*1000,
                iqr_ms=(values[3*len(values)//4]-values[len(values)//4])*1000,
                samples=len(values))


def memory(model, optimizer=None):
    result=dict(parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()))
    if optimizer is not None:
        result['optimizer_tensor_bytes']=sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    if sys.platform=='win32':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            result.update(process_working_set_bytes=counters.ws,process_peak_working_set_bytes=counters.peak_ws)
    return result


def save(path, model, optimizer, config, step, generator, extra=None):
    if isinstance(path,(str,Path)):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                    step=step,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                    task_rng=generator.get_state(),extra=extra or {}),path)


def restore(path,model,optimizer,generator,config):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['config'] != config:
        raise ValueError('Resume configuration must match the saved bounded run')
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
    generator.set_state(state['task_rng'])
    return state['step'],state['extra']


def write_json(path,result):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
