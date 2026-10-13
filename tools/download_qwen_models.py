"""Download the two Q8_0 Qwen artifacts used by the DS08 forward-parity test.

Weights stay out of the repository; they land under `.temp/qwen-corpus/` (git-ignored) and are verified
against the pinned size and SHA-256 before use. `qwen_forward_reference.py` produces the independent
float64 fixture from the same files.
"""
import hashlib
import urllib.request
from pathlib import Path

project = Path(__file__).resolve().parents[1]
temporary = (project / '.temp').resolve()
root = (temporary / 'qwen-corpus').resolve()
if not temporary.is_relative_to(project) or not root.is_relative_to(temporary):
    raise ValueError('Temporary corpus directory escapes the project')
root.mkdir(parents=True, exist_ok=True)

# name -> (repo, revision, remote name, size, sha256)
FILES = {
    'qwen2.5-coder-1.5b-q8_0.gguf': (
        'Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF', 'f86cb2c1fa58255f8052cc32aeede1b7482d4361',
        'qwen2.5-coder-1.5b-instruct-q8_0.gguf', 1894532160,
        '507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47'),
    'qwen3-4b-q8_0.gguf': (
        'Qwen/Qwen3-4B-GGUF', 'bc640142c66e1fdd12af0bd68f40445458f3869b',
        'Qwen3-4B-Q8_0.gguf', 4280404704,
        '8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300'),
}

for name, (repo, revision, remote, size, digest) in FILES.items():
    target = root / name
    if target.exists() and target.stat().st_size == size and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        print(target)
        continue
    partial = root / (name + '.part')
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{remote}'
    with urllib.request.urlopen(url) as response, partial.open('wb') as output:
        while chunk := response.read(1 << 20):
            output.write(chunk)
    if partial.stat().st_size != size or hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
        partial.unlink()
        raise ValueError(f'Verification failed for {name}')
    partial.replace(target)
    print(target)
