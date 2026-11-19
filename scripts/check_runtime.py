"""Preflight for a rented GPU instance; no LLM weights or responses are loaded."""
import json
import platform
import sys
import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check the Vast GPU assignment and host driver.')
    cuda = tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2]))
    report = dict(python=sys.version.split()[0], architecture=platform.machine(),
                  torch=torch.__version__, cuda=torch.version.cuda, gpus=[])
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if props.major >= 12 and cuda < (12, 8):
            raise SystemExit('RTX 50/Blackwell requires a CUDA 12.8+ PyTorch build.')
        # Actually launch a small kernel: version strings alone do not verify driver compatibility.
        with torch.cuda.device(i):
            x = torch.ones((32, 32), dtype=torch.bfloat16, device=f'cuda:{i}')
            value = (x @ x)[0, 0].item()
            torch.cuda.synchronize(i)
            if value != 32:
                raise RuntimeError('CUDA numerical preflight failed')
        report['gpus'].append(dict(id=i, name=props.name, capability=f'{props.major}.{props.minor}',
                                   memory_gib=round(props.total_memory/2**30, 1)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
