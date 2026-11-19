"""Build a small GitHub-ready source ZIP using an explicit allowlist."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    roots = ('llm_astar', 'scripts', 'configs', 'data', 'examples', 'tests')
    for name in roots:
        for path in sorted((ROOT/name).rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                if path.suffix in {'.py', '.sh', '.json', '.txt', '.md'} or path.name == 'LICENSE':
                    yield path
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and (path.name in {'.gitignore', 'run.py'} or path.name.startswith(('README', 'THIRD_PARTY', 'requirements'))):
            yield path
    for path in sorted((ROOT/'exp').glob('*/*')):
        if path.is_file() and not path.is_symlink() and path.suffix in {'.py', '.md'}:
            yield path
    yield ROOT/'models/llama3-download-manifest.json'


def main():
    output = ROOT/'dist/llm-astar-source.zip'
    output.parent.mkdir(exist_ok=True)
    files = list(source_files())
    for path in files:
        if not path.resolve().is_relative_to(ROOT) or path.stat().st_size > 10*1024*1024:
            raise ValueError(f'Unexpected source file: {path}')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path('llm-astar')/path.relative_to(ROOT))
    print(f'{output}: {len(files)} files, {output.stat().st_size:,} bytes; no model weights or experiment outputs')


if __name__ == '__main__':
    main()
