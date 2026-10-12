"""Extract pinned tokenizer metadata without weights; standard library only."""
import hashlib, struct
from pathlib import Path
source=Path('.temp/gguf-corpus/stories15M-q8_0.gguf').read_bytes()
assert hashlib.sha256(source).hexdigest()=='2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6'
pos=24

def read(fmt):
    global pos
    value=struct.unpack_from('<'+fmt,source,pos)[0];pos+=struct.calcsize('<'+fmt);return value

def string():
    global pos
    n=read('Q');v=source[pos:pos+n];pos+=n;return v

def value(t):
    if t==8: string()
    elif t==9:
        element=read('I');count=read('Q')
        for _ in range(count):value(element)
    else:read({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}[t])
entries=[]
for _ in range(struct.unpack_from('<Q',source,16)[0]):
    start=pos;key=string().decode();t=read('I');value(t)
    if key.startswith('tokenizer.') or key=='general.architecture':entries.append(source[start:pos])
def s(v):return struct.pack('<Q',len(v))+v
out=b'GGUF'+struct.pack('<IQQ',3,1,len(entries))+b''.join(entries)
out+=s(b'dummy')+struct.pack('<IQIQ',1,1,0,0)
out+=bytes((-len(out))%32)+bytes(32)
p=Path('fixtures/models/stories15m-q8-v1/tokenizer.gguf');p.write_bytes(out)
print(len(out),hashlib.sha256(out).hexdigest())
