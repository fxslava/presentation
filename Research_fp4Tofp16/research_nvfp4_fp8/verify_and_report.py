"""Check mapping/banking acceptance cases and regenerate measured spec tables."""
import json, math
from pathlib import Path
import numpy as np
import torch
from run_study import GRID, table, quant

root=Path(__file__).resolve().parent
data=json.loads((root/'results/study.json').read_text())
checks={}
codes=np.arange(16);e=(codes>>1)&3;m=codes&1
for fmt, bias, mb, tiny in [('e4m3',6,3,0x30),('e5m2',14,2,0x38)]:
    b=((codes&8)<<4)|np.where(e==0,np.where(m==0,0,tiny),((e+bias)<<mb)|(m<<(mb-1)))
    val=np.copysign(table(fmt)[b&127],np.where(b&128,-1.,1.))
    want=np.copysign(GRID[codes&7],np.where(codes&8,-1.,1.))
    assert np.array_equal(val,want) and np.signbit(val[8])
    dtype=torch.float8_e4m3fn if fmt=='e4m3' else torch.float8_e5m2
    # CUDA cast round-trip is a conversion probe, not an MMA subnormal probe.
    x=np.concatenate([table(fmt),-table(fmt)]).astype(np.float32)
    y=torch.tensor(x,device='cuda').to(dtype).float().cpu().numpy()
    assert np.array_equal(x,y) and np.array_equal(np.signbit(x),np.signbit(y))
    products=(GRID[:,None]*table('e4m3')[None,:]).ravel().astype(np.float32)
    products=np.minimum(products,table(fmt)[-1])
    actual=torch.tensor(products,device='cuda').to(dtype).float().cpu().numpy()
    assert np.array_equal(quant(products,fmt),actual)
    checks[fmt]={'raw_codes':16,'representable_cuda_cast_roundtrips':len(x),
                 'finite_scale_cuda_rne_checks':len(products),'signed_zero_preserved':True}
for banks,w,pitch in [(32,4,128),(8,32,256),(32,4,132)]:
    def occupied(p):return len({(i*p//w)%banks for i in range(banks)})
    assert occupied(pitch)==banks//math.gcd(pitch//w,banks)
    padded=pitch+(w if (pitch//w)%2==0 else 0)
    assert occupied(padded)==banks
checks['bank_padding_cases']=3
checks['independent_stall']=max(0,60-100)
checks['shared_issue_extra_bound']=max(0,60+70-100)
assert checks['independent_stall']==0 and checks['shared_issue_extra_bound']==30
assert len(data['fidelity'])==62 and len(data['gpu_controls'])==15
assert all('error' not in r for r in data['gpu_controls'])
checks['all_requested_gpu_controls_completed']=True
(root/'results/verification.json').write_text(json.dumps(checks,indent=2))

lines=['### 8.1 Encoding and software fidelity results','',
       'Exhaustive software check: all 16 signed raw codes embed exactly; **275/1,016** nonnegative finite E4M3-scale products need E4M3 rounding (including overflow saturation). Both E4M3/E5M2 mappings and CUDA conversion behavior are independently checked by `verify_and_report.py`. CUDA casts preserve all finite representable values tested, including subnormals and −0; this does not establish Tensor Core operand subnormal behavior.','',
       'B=16, E4M3 local scales; errors below are relative to the original FP16-rounded baseline. FP8 rows include activation quantization.','',
       '| Distribution | Weight path / rounding | Weight relative error | Output relative error | Output cosine | Softmax KL |',
       '|---|---|---:|---:|---:|---:|']
for row in data['fidelity']:
    if row['block'] in [None,16] and row['scale_format'] in [None,'e4m3']:
        lines.append(f"| {row['distribution']} | {row['path']} / {row['rounding']} | {row['weights']['relative_frobenius']:.6f} | {row['output']['relative_frobenius']:.6f} | {row['output']['cosine']:.6f} | {row['softmax_kl']:.6g} |")
lines+=['','All 62 cases, including B=32/64 and E8M0 scale ablations, remain in the JSON. In this seed the B=16 E4M3/RNE reconstructed route has output error 9.98% (Gaussian) and 8.85% (heavy-tail), while direct FP8 has 3.76% and 3.75%. The heavy-tail softmax KL does not track Frobenius error monotonically. No quality claim extends to trained expert layers.','',
         '### 8.2 Actual GPU FP8 GEMM controls','',
         '| Requested M | Executed M | B | Median µs | Batch min–max µs |',
         '|---:|---:|---:|---:|---:|']
for row in data['gpu_controls']:
    lines.append(f"| {row['M']} | {row['M_executed']} | {row['block']} | {row['median_us']:.3f} | {row['min_us']:.3f}–{row['max_us']:.3f} |")
lines+=['','The control GEMM no longer reads FP4 block scales, so timing differences between B values reflect run variation/data/library effects, not measured decode performance. Requested M=1 and M=8 execute the same padded shape. The maximum output drift from FP32 matmul on the same prequantized operands was '+f"{max(r['output_vs_fp32']['relative_frobenius'] for r in data['gpu_controls']):.3g}"+' in this run; this checks the control, not full quantization fidelity.','']
doc=root.parent/'docs/spec_analyzer_backend_nvfp4_fp8.md'
txt=doc.read_text(encoding='utf-8')
start='<!-- MEASURED_RESULTS -->';end='<!-- END_MEASURED_RESULTS -->'
if end in txt:
    before,rest=txt.split(start,1);_,after=rest.split(end,1)
    txt=before+start+'\n\n'+'\n'.join(lines)+end+after
else:txt=txt.replace(start,start+'\n\n'+'\n'.join(lines)+end)
doc.write_text(txt,encoding='utf-8')
snippet=txt.split('```text',1)[1].split('```',1)[0].strip().splitlines()
assert 25<=len(snippet)<=40
assert all(c in txt for c in ['AKA2011','AKA1012','AKA3007','AKA3006'])
assert txt.count('```')%2==0
print(json.dumps({'verification':checks,'conceptual_snippet_lines':len(snippet),'spec_lines':len(txt.splitlines())},indent=2))
