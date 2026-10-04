"""Generate isolated micro-tile copies of the existing prototypes."""
from pathlib import Path
root=Path(__file__).resolve().parent
src=root.parent
(root/'src').mkdir(exist_ok=True)
simd=(src/'simd_naive_unpack.cpp').read_text()
simd=simd.replace('static constexpr uint32_t AIV_TILE_ELEMS = 8192;', 'static constexpr uint32_t AIV_TILE_ELEMS = FAST_TILE;')
simd=simd.replace('static constexpr uint32_t AIV_TILE_OUT_BYTES', 'static constexpr uint32_t AIV_SCALE_COPY_BYTES = ((AIV_TILE_SCALES + 31) / 32) * 32;\nstatic constexpr uint32_t AIV_TILE_OUT_BYTES')
simd=simd.replace('AIV_QUE_DEPTH, AIV_TILE_SCALES)', 'AIV_QUE_DEPTH, AIV_SCALE_COPY_BYTES)')
simd=simd.replace('                 AIV_TILE_SCALES);','                 AIV_SCALE_COPY_BYTES);')
old='Cast(h, mag, RoundMode::CAST_NONE, n);              // 0..7 exact'
new='''Cast(h, mag, RoundMode::CAST_NONE, n);
#ifdef FAST_E2M1
        // Produce 4*q = {0,2,4,6,8,12,16,24}, then reuse scale s/4.
        Muls(h, h, static_cast<half>(2), n);
        CompareScalar(msk, h, static_cast<half>(8), CMPMODE::GT, n);
        Muls(t, h, static_cast<half>(2), n);
        Adds(t, t, static_cast<half>(-8), n);
        Select(h, msk, t, h, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        CompareScalar(msk, h, static_cast<half>(16), CMPMODE::GT, n);
        Duplicate(t, static_cast<half>(24), n);
        Select(h, msk, t, h, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Cast(t, codes, RoundMode::CAST_NONE, n);
        CompareScalar(msk, t, static_cast<half>(7), CMPMODE::GT, n);
#endif'''
assert old in simd
simd=simd.replace(old,new)
(root/'src/simd.cpp').write_text(simd)
simt=(src/'simt_direct_gm_unpack.cpp').read_text().replace('8192','FAST_TILE')
simt=simt.replace('int32_t signed_code = (code & 8)', '''int32_t magnitude = code & 7;
#ifdef FAST_E2M1
        magnitude = magnitude <= 4 ? 2*magnitude : (magnitude == 5 ? 12 : (magnitude == 6 ? 16 : 24));
#endif
        int32_t signed_code = (code & 8)''').replace('static_cast<int32_t>(code & 7)','magnitude')
(root/'src/simt.cpp').write_text(simt)
cube=(src/'cube_trick_fp4_to_fp16.cpp').read_text()
(root/'src/cube.cpp').write_text(cube)
harness=(src/'harness.cpp').read_text()
harness=harness.replace('s(n/FP4_BLOCK)', 's(((n/FP4_BLOCK+31)/32)*32)')
harness=harness.replace('int v=(code&7)*(code&8?-1:1);','''int mag=code&7;
#ifdef FAST_E2M1
      mag=mag<=4 ? 2*mag : (mag==5 ? 12 : (mag==6 ? 16 : 24));
#endif
      int v=mag*(code&8?-1:1);''')
harness=harness.replace('unsigned tile=i/8192, local=i%8192;', 'unsigned tile=i/FAST_TILE, local=i%FAST_TILE;')
harness=harness.replace('tile*8192+(local%2)*4096+local/2','tile*FAST_TILE+(local%2)*(FAST_TILE/2)+local/2')
harness=harness.replace('n%256!=0','n%256!=0 || n%FAST_TILE!=0 || n%FP4_BLOCK!=0')
harness=harness.replace('const unsigned tiles=n/256;', '''const unsigned tiles=n/256;
  printf("FORMAT=%s TILE=%u BLOCK=%u\\n",
#ifdef FAST_E2M1
         "E2M1",
#else
         "E1M2",
#endif
         unsigned(FAST_TILE),unsigned(FP4_BLOCK));''')
(root/'src/harness.cpp').write_text(harness)
print('Prepared independent micro-tile E2M1/E1M2 build inputs.')
