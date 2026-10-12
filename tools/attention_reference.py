import math,struct,json,pathlib
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
width=288
q=[f32((i%17-8)*.125) for i in range(width)]
k=[[f32(((t*13+i*7)%31-15)*.0625) for i in range(width)] for t in range(128)]
v=[[f32(((t*5+i*11)%37-18)*.125) for i in range(width)] for t in range(128)]
cases=[]
for count in [1,2,32,64,128]:
 out=[]
 for head in range(0,width,48):
  score=[sum(q[head+j]*k[t][head+j] for j in range(48))/math.sqrt(48) for t in range(count)]
  weights=[math.exp(x-max(score)) for x in score];total=sum(weights)
  out.extend(sum(weights[t]/total*v[t][head+j] for t in range(count)) for j in range(48))
 cases.append({'count':count,'expected':out})
pathlib.Path('fixtures/operators/attention-boundaries-v1.json').write_text(json.dumps({'reference':'Python binary64 scalar dot/max-subtracted softmax/weighted sum; inputs rounded to F32; deterministic formulas in VulkanAttentionTests','cases':cases},indent=2)+'\n')
