"""Reproducible software fidelity + measured materialized FP8 GPU controls.
Not a fused GEMM implementation. Run from repository root with Python/NumPy/Torch.
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np

GRID = np.array([0,.5,1,1.5,2,3,4,6], dtype=np.float32)

def table(fmt):
    b = np.arange(128)
    mb, bias = (3,7) if fmt == 'e4m3' else (2,15)
    e, m = b >> mb, b & ((1<<mb)-1)
    v = np.where(e == 0, m / (1<<mb) * 2.**(1-bias),
                 (1+m/(1<<mb))*2.**(e-bias))
    return v[:127] if fmt == 'e4m3' else v[:124]

def quant(x, fmt='e4m3', mode='rne', rng=None):
    t = table(fmt); a = np.abs(x)
    hi = np.minimum(np.searchsorted(t,a),len(t)-1); lo=np.maximum(hi-1,0)
    dl, dh = a-t[lo], t[hi]-a
    if mode == 'stochastic':
        p=np.divide(dl,t[hi]-t[lo],out=np.zeros_like(a),where=hi!=lo)
        pick = rng.random(a.shape) < np.clip(p,0,1)
    else:
        pick=(dh<dl)|((dh==dl)&((hi&1)==0))
    return np.copysign(t[np.where(pick,hi,lo)],x).astype(np.float32)

def nvfp4(w, block, scale_format):
    g=w.reshape(-1,block)
    ideal=np.max(np.abs(g),axis=1)/6
    # FP32 tensor factor makes local scale range fit the scale format.
    global_scale=float(max(np.max(ideal)/448, np.finfo(np.float32).tiny)) if scale_format=='e4m3' else 1.
    if scale_format=='e4m3': s=quant(ideal/global_scale)*global_scale
    else: s=np.exp2(np.clip(np.ceil(np.log2(np.maximum(ideal,2.**-127))),-127,127)).astype(np.float32)
    s=np.where(ideal==0,1,s)
    z=np.abs(g)/s[:,None]
    hi=np.minimum(np.searchsorted(GRID,z),7); lo=np.maximum(hi-1,0)
    pick=(GRID[hi]-z < z-GRID[lo])|((GRID[hi]-z == z-GRID[lo])&((hi&1)==0))
    q=np.copysign(GRID[np.where(pick,hi,lo)],g)
    return (q*s[:,None]).reshape(w.shape), global_scale

def metrics(ref, got):
    r=ref.astype(np.float64).ravel(); g=got.astype(np.float64).ravel()
    return {'relative_frobenius':float(np.linalg.norm(r-g)/max(np.linalg.norm(r),1e-300)),
            'cosine':float(np.dot(r,g)/max(np.linalg.norm(r)*np.linalg.norm(g),1e-300))}

def kl(ref,got):
    # KL between per-token softmax distributions; these are synthetic logits.
    r=ref.astype(np.float64);g=got.astype(np.float64)
    lr=r-r.max(1,keepdims=True); lg=g-g.max(1,keepdims=True)
    lr-=np.log(np.exp(lr).sum(1,keepdims=True));lg-=np.log(np.exp(lg).sum(1,keepdims=True))
    return float(np.mean(np.sum(np.exp(lr)*(lr-lg),axis=1)))

def exhaustive():
    codes=np.arange(16); e=(codes>>1)&3; m=codes&1
    mag=np.where(e==0,np.where(m==0,0,0x30),((e+6)<<3)|(m<<2))
    bits=((codes&8)<<4)|mag
    decoded=np.copysign(table('e4m3')[bits&127],np.where(bits&128,-1.,1.))
    expected=np.copysign(GRID[codes&7],np.where(codes&8,-1.,1.))
    assert np.array_equal(decoded,expected)
    assert np.signbit(decoded[8]) and np.signbit(quant(np.array([-0.]))[0])
    values=(GRID[:,None]*table('e4m3')[None,:]).ravel()
    got=quant(values)
    # Confirm RNE tie-to-even with independent exhaustive distances.
    for x,y in zip(values,got):
        dist=np.abs(table('e4m3')-x); ids=np.flatnonzero(dist==dist.min())
        i=next((int(i) for i in ids if i%2==0),int(ids[0]))
        assert y==table('e4m3')[i]
    return {'raw_mapping_codes':16,'e4m3_scale_products':len(values),
            'nonexact_scale_products':int(np.count_nonzero(got!=values)),
            'signed_zero_preserved':True,'e4m3_min_subnormal':float(table('e4m3')[1])}

def timing(fn,torch,reps):
    for _ in range(5): fn()
    torch.cuda.synchronize(); samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):fn()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/reps)
    return {'median_us':float(np.median(samples)),'min_us':min(samples),'max_us':max(samples)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--out',default='research_nvfp4_fp8/results');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261004)
    k=n=4096 if args.full else 512
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    result={'seed':20261004,'N':n,'K':k,'python':platform.python_version(), 'torch':torch.__version__,
            'device':torch.cuda.get_device_name(),'exhaustive':exhaustive(),'fidelity':[],'gpu_controls':[]}
    for dist in ['gaussian','heavy_tail']:
        w=rng.normal(size=(n,k)).astype(np.float32)/np.sqrt(k)
        x=rng.normal(size=(32,k)).astype(np.float32)
        if dist=='heavy_tail':
            w[rng.random(w.shape)<.002]*=32
            x[rng.random(x.shape)<.005]*=16
        w=w.astype(np.float16).astype(np.float32);x=x.astype(np.float16).astype(np.float32)
        xd=torch.tensor(x,device='cuda');wd=torch.tensor(w,device='cuda')
        ref=(xd@wd.T).cpu().numpy()
        # Explicit global factors retained outside FP8 data.
        ax=max(float(np.max(np.abs(x)))/448,1e-30);bw=max(float(np.max(np.abs(w)))/448,1e-30)
        xq=quant(x/ax)*ax
        direct=quant(w/bw)*bw
        variants=[('direct_fp8',None,None,'rne',direct)]
        for block in [16,32,64]:
            for sf in ['e4m3','e8m0']:
                dq,glob=nvfp4(w,block,sf)
                variants.append(('nvfp4_reconstructed',block,sf,'none',dq))
                for fmt in ['e4m3','e5m2']:
                    maximum=448 if fmt=='e4m3' else 57344
                    factor=max(float(np.max(np.abs(dq)))/maximum,1e-30)
                    for mode in ['rne','stochastic']:
                        variants.append(('nvfp4_to_'+fmt,block,sf,mode,quant(dq/factor,fmt,mode,rng)*factor))
        for name,block,sf,mode,approx in variants:
            xx=xd if name=='nvfp4_reconstructed' else torch.tensor(xq,device='cuda')
            y=(xx@torch.tensor(approx,device='cuda').T).cpu().numpy()
            result['fidelity'].append({'distribution':dist,'path':name,'block':block,'scale_format':sf,'rounding':mode,
                'weights':metrics(w,approx),'output':metrics(ref,y),'softmax_kl':kl(ref,y)})
        del xd,wd
    # Actual FP8 Tensor Core control via Torch/cuBLAS. M<16 padded to 16.
    w=rng.normal(size=(n,k)).astype(np.float32)/np.sqrt(k)
    for block in [16,32,64]:
        dq,_=nvfp4(w,block,'e4m3');bw=max(float(np.max(np.abs(dq)))/448,1e-30)
        b=torch.tensor(dq/bw,device='cuda').to(torch.float8_e4m3fn).T
        sb=torch.tensor(bw,device='cuda');sa=torch.ones((),device='cuda')
        for m in [1,8,32,128,2048]:
            mp=max(16,m);a=torch.randn(mp,k,device='cuda').to(torch.float8_e4m3fn)
            fn=lambda:torch._scaled_mm(a,b,scale_a=sa,scale_b=sb,out_dtype=torch.float32,use_fast_accum=False)
            try:
                t=timing(fn,torch,20)
                y=fn();gold=a.float()@b.float()*bw
                result['gpu_controls'].append({'M':m,'M_executed':mp,'block':block,'kind':'materialized_fp8_gemm_only',**t,
                                               'output_vs_fp32':metrics(gold.cpu().numpy(),y.cpu().numpy())})
            except Exception as ex: result['gpu_controls'].append({'M':m,'block':block,'error':str(ex)})
    (out/'study.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(out/'study.json'),'exhaustive':result['exhaustive'],'fidelity_rows':len(result['fidelity']),'gpu_rows':len(result['gpu_controls'])},indent=2))

if __name__=='__main__':main()
