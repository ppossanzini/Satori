"""Regenerate embedded Vulkan shader artifacts using the pinned compiler tools."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
compiler = subprocess.check_output(['glslangValidator', '--version'], text=True)
validator = subprocess.check_output(['spirv-val', '--version'], text=True)
if '16.6.0' not in compiler or '2026.4' not in validator:
    raise SystemExit('Expected glslang 16.6.0 and SPIRV-Tools 2026.4.')
for shader in sorted((root / 'src/Satori.Backends.Vulkan/Shaders').glob('*.comp')):
    binary = shader.with_suffix('.spv')
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.1', '-S', 'comp', '-e', 'main', '-o', str(binary), str(shader)], check=True)
    subprocess.run(['spirv-val', '--target-env', 'vulkan1.1', str(binary)], check=True)
    q8 = shader.stem == 'q8_matvec'
    bindings = {'0':'readonly packed Q8_0 uint words', '1':'readonly FP32 input', '2':'writeonly FP32 output'} if q8 else {'0':'readonly FP32 input', '1':'writeonly FP32 output'}
    push = {'rows':{'offset':0,'bytes':4}, 'columns':{'offset':4,'bytes':4}} if q8 else {'count':{'offset':0,'bytes':4}}
    manifest = dict(compiler=compiler.strip(), validator=validator.strip(), target='vulkan1.1', spirv='1.3', entry='main', local_size=[64,1,1], bindings=bindings, push_constants=push, source_sha256=hashlib.sha256(shader.read_bytes()).hexdigest(), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), compile_command=f'glslangValidator -V --target-env vulkan1.1 -S comp -e main -o {binary.name} {shader.name}', validate_command=f'spirv-val --target-env vulkan1.1 {binary.name}')
    binary.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
