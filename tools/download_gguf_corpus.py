"""Download the fixed public weight corpus into the project's temporary workspace."""
import hashlib
import urllib.request
from pathlib import Path

project = Path(__file__).resolve().parents[1]
temporary = (project / '.temp').resolve()
root = (temporary / 'gguf-corpus').resolve()
if not temporary.is_relative_to(project) or not root.is_relative_to(temporary):
    raise ValueError('Temporary corpus directory escapes the project')
root.mkdir(parents=True, exist_ok=True)
revision = '99dd1a73db5a37100bd4ae633f4cfce6560e1567'
files = {
    'stories15M-q8_0.gguf': (26671328, '2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6'),
    'stories15M-q4_0.gguf': (19077344, '6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04'),
}
for name, (size, digest) in files.items():
    target = root / name
    if target.exists() and target.stat().st_size == size and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        print(target)
        continue
    partial = root / (name + '.part')
    try:
        url = f'https://huggingface.co/ggml-org/tiny-llamas/resolve/{revision}/{name}'
        with urllib.request.urlopen(url) as response, partial.open('wb') as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > size:
                    raise ValueError('Unexpected corpus file size')
                output.write(chunk)
        if partial.stat().st_size != size or hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
            raise ValueError('Corpus checksum mismatch')
        partial.replace(target)
        print(target)
    finally:
        partial.unlink(missing_ok=True)
