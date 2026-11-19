"""Atomic experiment records and portable result archives (never model weights)."""
import json
import os
from pathlib import Path
import tempfile
import zipfile


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def bundle_results(output):
    """Allowlist experiment artifacts; exclude symlinks, checkpoints and credentials."""
    output = Path(output).resolve()
    destination = Path(str(output) + '.zip')
    temporary = Path(str(destination) + '.tmp')
    root_files = {'results.jsonl', 'summary.json', 'table1.csv', 'manifest.json',
                  'status.json', 'runtime.json', 'baselines.json', 'experiment-code.json', 'input.json'}
    subfolders = {'jobs', 'responses', 'logs'}
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(output.rglob('*')):
                rel = path.relative_to(output)
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p != output.parent):
                    continue
                allowed = (len(rel.parts) == 1 and
                           (path.name in root_files or path.name.startswith(('run-', 'environment-'))))
                allowed |= len(rel.parts) > 1 and rel.parts[0] in subfolders
                if allowed and path.suffix in {'.json', '.jsonl', '.csv', '.log'}:
                    archive.write(path, Path(output.name) / rel)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
