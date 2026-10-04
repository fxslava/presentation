#include <acl/acl.h>
#ifndef RAW_BINARY
#include "aclrtlaunch_naive_aiv_fp4_to_fp16.h"
#include "aclrtlaunch_cube_trick_fp4_to_fp16.h"
#endif
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifndef FP4_BLOCK
#define FP4_BLOCK 16
#endif
#define CHECK(x) do { auto e=(x); if(e) {fprintf(stderr,"%s failed: %u\n",#x,unsigned(e)); return 2;} } while(0)
#ifdef RAW_BINARY
uint32_t RawLaunch(const char*path,const char*name,aclrtStream stream,void**args,size_t count) {
  FILE*f=fopen(path,"rb"); if(!f) return 2;
  fseek(f,0,SEEK_END);long size=ftell(f); rewind(f);
  std::vector<uint8_t> data(size);if(fread(data.data(),1,size,f)!=size) {fclose(f);return 2;}fclose(f);
  aclrtBinary binary=aclrtCreateBinary(data.data(),data.size()); if(!binary) return 2;
  aclrtBinHandle handle=nullptr;CHECK(aclrtBinaryLoad(binary,&handle));
  aclrtFuncHandle function=nullptr;CHECK(aclrtBinaryGetFunction(handle,name,&function));
  void*deviceArgs=nullptr;CHECK(aclrtMalloc(&deviceArgs,count*sizeof(void*),ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(deviceArgs,count*sizeof(void*),args,count*sizeof(void*),ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtLaunchKernel(function,1,deviceArgs,count*sizeof(void*),stream));
  CHECK(aclrtSynchronizeStream(stream));
  aclrtFree(deviceArgs);aclrtBinaryUnLoad(handle);aclrtDestroyBinary(binary);
  return 0;
}
#endif
int main(int argc,char**argv) {
  const bool cube=argc>1 && argv[1][0]=='B';
#ifdef RAW_BINARY
  if(argc<3) {fprintf(stderr,"Usage: raw_harness A|B kernel.o\n");return 2;}
#endif
  const unsigned n=getenv("FP4_ELEMENTS") ? unsigned(strtoul(getenv("FP4_ELEMENTS"),nullptr,10)) : 8192;
  if(n==0 || n%256!=0) {fprintf(stderr,"FP4_ELEMENTS must be a positive multiple of 256\n");return 2;}
  const unsigned tiles=n/256;
  std::vector<uint8_t> w(n/2),s(n/FP4_BLOCK),cs(tiles*32);
  std::vector<int8_t> a(tiles*512),b(tiles*512);
  std::vector<uint16_t> ref(n),out(n);
  for(unsigned i=0;i<n;++i) {
    unsigned code=(i*13+i/16)%16;
    w[i/2]|=code<<((i%2)*4);
  }
  for(unsigned q=0;q<n/FP4_BLOCK;++q) {
    unsigned mant=q%8,exp=1+(q/8)%15;
    if(exp==15&&mant==7) mant=6;
    s[q]=(exp<<3)|mant;
  }
  for(unsigned g=0;g<n/16;++g) {
    uint8_t sb=s[(g*16)/FP4_BLOCK];
    unsigned mant=sb&7,exp=sb>>3;
    unsigned t=g/16,r=g%16;
    cs[t*32+r]=sb;
    int av0=mant==0?8:(mant<5?16:24),av1=16+2*mant-av0;
    a[t*512+r*32+r]=av0;
    a[t*512+r*32+16+r]=av1;
    for(unsigned j=0;j<16;++j) {
      unsigned i=g*16+j,code=(w[i/2]>>((i%2)*4))&15;
      int v=(code&7)*(code&8?-1:1);
      b[t*512+j*32+r]=v;
      b[t*512+j*32+16+r]=v;
      float value=v*0.25f*std::ldexp(1.0f+mant/8.0f,int(exp)-7);
      if(v==0) ref[i]=0;
      else {
        // All generated nonzero products are exactly representable normal halves.
        int exponent=0;float fraction=std::frexp(std::fabs(value),&exponent);
        unsigned fractionBits=unsigned((fraction*2-1)*1024);
        ref[i]=uint16_t((v<0?0x8000:0)|((exponent+14)<<10)|fractionBits);
      }
    }
  }
#if defined(SIMT_SWITCH_PROBE) || defined(SCALAR_SWITCH_PROBE)
  std::fill(ref.begin(), ref.end(), 0);
#endif
  CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
  aclrtStream stream; CHECK(aclrtCreateStream(&stream));
  std::vector<void*> alloc;
  auto upload=[&](const void*src,size_t size)->void* {
    void*p=nullptr; auto e=aclrtMalloc(&p,size,ACL_MEM_MALLOC_HUGE_FIRST);
    if(!e&&src) e=aclrtMemcpy(p,size,src,size,ACL_MEMCPY_HOST_TO_DEVICE);
    if(e) {fprintf(stderr,"allocation/upload failed %u\n",unsigned(e));exit(2);}
    alloc.push_back(p);return p;
  };
  uint32_t tiling[8]={cube?tiles:n};
  void*dt=upload(tiling,sizeof tiling),*dout=upload(out.data(),n*2);
  if(cube) {
#ifdef SIMT_HYBRID
    void*dw=upload(w.data(),w.size()),*ds=upload(s.data(),s.size());
    void*da=upload(nullptr,a.size()),*db=upload(nullptr,b.size()),*dc=upload(nullptr,n*2);
    void*args[]={dw,ds,da,db,dc,dout,dt};
    CHECK(RawLaunch(argv[2],"simt_cube_hybrid",stream,args,7));
#else
    void*da=upload(a.data(),a.size()),*db=upload(b.data(),b.size()),*ds=upload(cs.data(),cs.size()),*dc=upload(nullptr,getenv("CONTROL_STAGE32")?2048:1024);
#ifdef RAW_BINARY
    void*args[]={da,db,ds,dc,dout,dt};
    CHECK(RawLaunch(argv[2],"cube_trick_fp4_to_fp16",stream,args,6));
#else
    CHECK(aclrtlaunch_cube_trick_fp4_to_fp16(1,stream,da,db,ds,dc,dout,dt));
#endif
#endif
  } else {
    void*dw=upload(w.data(),w.size()),*ds=upload(s.data(),s.size());
#ifdef RAW_BINARY
    void*args[]={dw,ds,dout,dt};
#ifdef SIMT_SWITCH_PROBE
    CHECK(RawLaunch(argv[2],"simt_switch_probe",stream,args,4));
#elif defined(SCALAR_SWITCH_PROBE)
    CHECK(RawLaunch(argv[2],"scalar_switch_probe",stream,args,4));
#elif defined(SIMT_DIRECT)
    CHECK(RawLaunch(argv[2],"simt_direct_gm_unpack",stream,args,4));
#else
    CHECK(RawLaunch(argv[2],"naive_aiv_fp4_to_fp16",stream,args,4));
#endif
#else
    CHECK(aclrtlaunch_naive_aiv_fp4_to_fp16(1,stream,dw,ds,dout,dt));
#endif
  }
  CHECK(aclrtSynchronizeStream(stream));
  CHECK(aclrtMemcpy(out.data(),n*2,dout,n*2,ACL_MEMCPY_DEVICE_TO_HOST));
  unsigned bad=0;
  for(unsigned i=0;i<n;++i) {
    unsigned tile=i/8192, local=i%8192;
    unsigned oi=cube?i:(tile*8192+(local%2)*4096+local/2);
#if defined(SIMT_DIRECT) || defined(SIMT_SWITCH_PROBE) || defined(SCALAR_SWITCH_PROBE)
    oi=i;
#endif
    if(out[oi]!=ref[i]) {if(bad<8) printf("mismatch %u expected=%04x got=%04x\n",i,ref[i],out[oi]);++bad;}
  }
  printf("Kernel %c elements=%u bit_mismatches=%u RESULT=%s\n",cube?'B':'A',n,bad,bad?"FAIL":"PASS");
  for(void*p:alloc) aclrtFree(p);
  aclrtDestroyStream(stream);aclrtResetDevice(0);aclFinalize();
  return bad?1:0;
}
