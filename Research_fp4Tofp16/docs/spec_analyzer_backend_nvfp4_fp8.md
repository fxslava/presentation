# NVFP4 → FP8: analyzer verification and PipeSim calibration specification

Status: proposed backend contract; numerical software experiments and local FP8 GEMM controls accompany this document. No standalone CUDA/CUTLASS kernel is implemented. Backend source changes are outside this deliverable.

## 1. Scope, evidence, and format contract

The objective is to model a packed FP4 weight stream, its conversion to FP8 in on-chip storage, and a ragged expert contraction without charging for a nonexistent FP16 DRAM tensor. The analyzer must establish numerical legality, allocation feasibility, memory geometry, and synchronization before PipeSim assigns overlap.

**Evidence classes:** `software_measured` means reference-format enumeration or numerical emulation; `gpu_measured` means an actual GPU operation; `analytical` means the formulas below; `uncalibrated` means an unknown hardware coefficient. Never render an uncalibrated cycle count as measured. Instruction counts are exact only for a specified lowered instruction sequence and active-lane map. ISA semantics do not publish universal ALU instruction latencies.

Local environment: RTX 5070, SM120, 12,227 MiB reported memory, driver 595.79, CUDA toolkit 13.2, PyTorch 2.13.0.dev20260516+cu130. This is neither Hopper SM90 nor datacenter Blackwell SM100. FP8 Tensor Core controls are executed through `torch._scaled_mm`; they do not measure fused conversion, TMA, WGMMA, or NPU hardware. The synthetic fidelity study uses N=K=4096. Its CPU format emulation and GPU FP32 contractions are distinct from the measured FP8 controls. Results and command are in §8.

### 1.1 Required mathematical and storage metadata

For expert e, output channel n, and reduction index k:

\[
W_{e,n,k}=g_e\,s_{e,n,\lfloor k/B\rfloor}\,q_{e,n,k},\qquad
Y_{e,m,n}=\sum_k A_{e,m,k}W_{e,n,k}.
\]

`g_e` denotes the **dequantization multiplier**, not the reciprocal quantizer scale. State the direction explicitly in IR. Standard NVFP4 is E2M1 with B=16, an E4M3 block multiplier, and an FP32 tensor multiplier. B=32/64 with E4M3 and B=16/64 with E8M0 are experimental generalized formats, not automatically NVFP4. MXFP4 uses B=32 and E8M0. Require `format_family`, payload encoding, nibble order, scale type and direction, grouping axis, B, tensor factor, rounding, zero/NaN policy, and contraction operand layout. See [Transformer Engine NVFP4 format](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html).

The supported test contract uses finite nonnegative local scales, signed E2M1 payloads, saturation on FP8 overflow, round-to-nearest-even (RNE), and signed-zero preservation. Zero scales are legal if explicitly handled; negative/nonfinite scales require a separate contract. E8M0 has exponent bias 127; finite codes 0…254 represent 2^(code−127), and 255 is NaN. A zero block needs a defined scale policy because E8M0 has no zero code.

### 1.2 Comparison of operational paths

| Path | Legal implementation shape | Main modeled resources | Applicability and limitation |
|---|---|---|---|
| A: fused conversion | Decode at contraction ingress, directly to operand fragments or shared/UB tiles | Integer/vector decode, scale arithmetic, SRAM writes/reads, live registers, contraction dispatch | Starting candidate for small M; shared operands still need a verified layout and publication edge |
| B: direct mapping | Exact raw E2M1→FP8 byte map; guarded exponent translation for power-of-two scaling | Bitfield/select instructions; scale mantissa multiply and conversion when needed | A conversion technique used within A/C, not a distinct GEMM schedule; arbitrary E4M3 scale cannot be handled by exponent addition alone |
| C: producer/consumer | Separate copy, decode, and contraction actors using versioned ring slots | Same decode work as A, plus barrier/dispatch costs and deeper live SRAM | Candidate when measured conversion service exceeds contraction slack; separation does not remove a globally shared issue bottleneck |
| D: grouped expert GEMM | Persistent or grouped scheduling, scale placement justified algebraically | Routing descriptors, gather/scatter, tails, expert weight reuse, scale arithmetic | Compatible with A/C; one epilogue alpha is legal only for factors independent of reduction k |

Recommended **hypothesis to calibrate**, not a measured winner: grouped scheduling plus A/B at low M, and grouped scheduling plus C where conversion and contraction can overlap on distinct resources. Native Blackwell block-scaled FP4 GEMM is a useful external control but violates the requested FP8 contraction constraint. Hopper WGMMA uses its documented register/shared operand forms; shared FP8 operands do not become register operands by assumption. Blackwell SM100 `tcgen05` and SM120 `mma` need separate profiles. CUTLASS provides [SM120 examples](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) and a [Hopper mixed-input example](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm); their existing kernels are references, not implementations of this proposal.

## 2. Mathematical formulation and bit-level mapping

Let nibble c have s=(c≫3)&1, e=(c≫1)&3, m=c&1. E2M1 is finite-only:

\[
q(c)=(-1)^s\begin{cases}m/2&e=0,\\2^{e-1}(1+m/2)&e>0.\end{cases}
\]

Magnitude codes 0…7 represent {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The normal-only equation in the initial research brief does not apply at e=0. FP8 raw encoding is exactly:

\[
b_{43}=(s\ll7)\;|\;\begin{cases}0&e=0,m=0\\0x30&e=0,m=1\\((e+6)\ll3)|(m\ll2)&e>0,\end{cases}
\]
\[
b_{52}=(s\ll7)\;|\;\begin{cases}0&e=0,m=0\\0x38&e=0,m=1\\((e+14)\ll2)|(m\ll1)&e>0.\end{cases}
\]

Both embeddings are exact and preserve ±0. E4M3 has min subnormal 2^-9, min normal 2^-6, max finite 448, and NaN magnitude 0x7f; exponent 15 is not universally reserved. E5M2 has min subnormal 2^-16, min normal 2^-14, max finite 57344, and IEEE-style infinity/NaN encodings. A saturating conversion must not invent infinity in E4M3. Format and conversion behavior are specified by [PTX floating-point types and conversion instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/) and [CUDA FP8 conversions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html).

### 2.1 Two-scale transformation and exactness conditions

Choose a retained positive FP32 output multiplier h. FP8 payload is

\[
r=Q_{FP8}(q\,s\,g/h),\qquad\widehat W=h\,r.
\]

For a positive power-of-two effective factor 2^d, add d to the **FP8 exponent field only after normalizing raw q**, preserving the sign and mantissa. This is exact only when the result remains representable; transitions to subnormal require shifts and RNE, transitions to overflow require policy handling, and zeros must bypass exponent addition. An integer add across a packed byte without per-byte guards can carry into neighboring values or the sign.

For E4M3 scale s=2^a(1+j/8), the mantissa product is (1+m/2)(1+j/8), which may need more than three fraction bits. Example: q=1.5, s=1.125 gives 1.6875; RNE E4M3 gives 1.75. No bitfield permutation preserves every such product. A raw embedding remains exact; the reconstructed embedding is generally rounded. E8M0 scaling can avoid mantissa multiplication, but a non-power-of-two global g/h reintroduces it. Per-block adaptive h cannot be used with an ordinary tensor-wide alpha unless the contraction supports the corresponding block scale mechanism.

For retained activation multiplier a_e and weight multiplier h_e, ordinary GEMM computes a_e h_e Σ Q_A Q_W. A column scale h_{e,n} can be applied in a per-column epilogue if independent of k. Otherwise:

\[
Y_{mn}=g\sum_b s_{n,b}\left(\sum_{k\in b}A_{mk}q_{nk}\right),
\]

which is a blockwise partial-sum contraction, not one final alpha. Counterexample: A=[1,1], q=[1,1], scales [1,2] and A=[1,0] require alpha 1.5 and 1 respectively. Reject a universal post-sum scale rewrite. Gates/SwiGLU need additional end-to-end validation; the synthetic linear benchmark cannot establish MoE model accuracy.

### 2.2 Compact conceptual register extraction (32 lines)

The following is pseudocode with explicit scalar byte lanes; lowering should pack independent values, use shift/mask/select or `lop3`/`prmt` when profitable, and count the resulting native instructions. It is not a standalone kernel or a promise that the compiler emits byte SIMD.

```text
// Input: packed u32 contains eight E2M1 nibbles, low nibble first.
// Output: two u32 words contain eight exact raw E4M3 values.
function map_e2m1x8_to_e4m3x8(packed):
    out_lo = 0
    out_hi = 0
    for i in static_unroll(0..7):
        nib = (packed >> (4*i)) & 0xF
        sign = (nib & 8) << 4
        exp = (nib >> 1) & 3
        man = nib & 1
        normal = ((exp + 6) << 3) | (man << 2)
        tiny = select(man != 0, 0x30, 0x00)
        mag = select(exp != 0, normal, tiny)
        byte = sign | mag
        if i < 4:
            out_lo |= byte << (8*i)
        else:
            out_hi |= byte << (8*(i-4))
    return out_lo, out_hi
// Scaling is a separate operation, not hidden in the bit embedding.
function reconstruct(raw_byte, local_scale, global_over_h):
    if (raw_byte & 0x7F) == 0:
        return raw_byte             // preserve signed zero for finite scale
    if effective_scale_is_pow2_and_result_is_normal:
        return guarded_exponent_shift(raw_byte, effective_exponent)
    x = decode_e4m3_to_fp32(raw_byte)
    x = x * local_scale             // include mantissa; shared per B values
    x = x * global_over_h
    return cvt_rne_satfinite_e4m3(x)
// Publish decoded SRAM writes before issuing the consumer contraction.
// Return ring-slot ownership only after asynchronous operand reads finish.
// Lowered opcode mix and dependency DAG supply the issue-cost model.
```

Stochastic rounding chooses neighboring representable values with p_upper=(x−lower)/(upper−lower); it is unbiased before clipping. It adds RNG state, issue demand, and register lifetime. Neither unbiasedness nor E5M2's wider range implies a smaller error on one realization. RNE is the deterministic default. Treat FTZ and zero canonicalization as separate declared transformations; collect underflow/saturation counts and compare actual device conversions to the reference before asserting bit equivalence.

## 3. Analytical cost models for PipeSim `evaluator.py`

### 3.1 Replace element-count proxies with packed transactions and instruction groups

Required stage IR: `PackedCopy → Decode → ScaledConvert → Publish → OperandTransfer → Contract → Release`, with resource labels, dependency edges, shapes, active lanes, ring depth, operation type, and evidence provenance. These stage names are **proposed**, not existing APIs. Preserve packed and unpacked buffers as distinct objects.

Let E be useful elements per tile, b_in=4, b_out=8, L active hardware lanes, R register bits per lane, and p=⌊R/b_in⌋ packed values handled per input register (R=32 gives p=8). For coalesced mapping, instruction groups and local scale groups are:

\[
G=\left\lceil E/(Lp)\right\rceil,\quad S=\left\lceil E/B\right\rceil,\quad
D_{in}=\left\lceil Eb_{in}/8\right\rceil,\quad D_{out}=\left\lceil Eb_{out}/8\right\rceil.
\]

Use executed padded E_exec and actual active-lane masks for work, useful E for efficiency. If rows are independently nibble packed, use Σ_rows ⌈K_row/2⌉ rather than ⌈Σ_rows K_row/2⌉. Separate scale traffic S·b_scale/8, tensor-scale metadata, alignment, and burst rounding. Do not halve an already scheduled int8 MTE stage after evaluation: packed bytes affect overlap, bottleneck selection, capacity, and number of transactions before scheduling.

For opcode class j (extract, Boolean, select, permute, scale decode, multiply, FP8 convert, address, mask, branch, publish), exact sequence demand is:

\[
n_j=G c_{j,pack}+G_s c_{j,scale}+n_{j,tail}+n_{j,scalar},
\]

where G_s counts actual vector/warp-scale groups, not necessarily S/L; block broadcast, cross-lane shuffles, and strided scales determine it. The c coefficients are counts from lowering or compiler/SASS inspection. p=8 does **not** imply eight scaled products occur in one instruction. Ascend vector counts derive from supported vector lane width and repeat semantics, not CUDA warp size.

For resource r, incidence a_{rj}, and capacity I_r issue slots/cycle:

\[
C_r=\left\lceil\frac{\sum_j a_{rj}n_j}{I_r}\right\rceil,
\quad C_{mem}=\left\lceil\frac{D_{in}+D_{scales}+D_{out}}{BW_{local}}\right\rceil,
\quad C_{dep}=\max_{\pi\in DAG}\sum_{v\in\pi}\lambda_v.
\]

Memory demand must be split by port/channel when reads and writes differ. λ is dependency latency; issue service is not dependency latency. A conditional closed form for independent resources with ideal overlap is:

\[
T_{unpack}^{LB}=\max(C_{dep},\max_r C_r,C_{mem}).
\]

Under a single serialized issue resource, no memory stalls, and a fully specified in-order instruction DAG, an exact recurrence is:

\[
t_v=\max(\max_{u\to v}(t_u+\lambda_u),\,available_{r(v)}),\qquad
available_{r(v)}\leftarrow t_v+\tau_v,
\quad T=\max_v(t_v+\lambda_v).
\]

For multi-resource instructions reserve **all** required resources; evaluate a declared deterministic dispatch policy or enumerate feasible schedules. Recurrence gives exact time within that policy/model, not universal silicon timing. Report an ideal-overlap lower bound and a dependency-respecting serialized upper bound until capacities, latencies, dispatch sharing, and memory paths are calibrated. Unknown coefficients must remain symbolic or be explicit intervals.

**Worked issue example (assumed coefficients, not silicon calibration):** E=4096, L=32, p=8 gives G=16. A lowered mapping assumed to cost 12 integer issue instructions/group plus 8 address/mask instructions requires 200 warp instructions. At one such issue slot/cycle, C_INT=200 cycles, not 4096 scalar iterations. Twelve is an illustrative coefficient, not the count of the pseudocode above. If scale products add 16 native instructions/group, add 256 slots to their actual resource(s). A profile that shares their dispatcher with INT must include the aggregate demand.

### 3.2 Scalar contention and dispatch accounting

Required simplified exposed-tail formula:

\[
T_{stall}=\max(0,T_{scalar\_addr}-T_{compute}).
\]

This equation is valid when scalar/address work starts concurrently with compute, is a separate resource, and only its uncovered tail delays the next tile. Then T_tile=T_compute+T_stall=max(T_compute,T_scalar_addr). Include mask, loop, descriptor, routing, and tail control in T_scalar_addr. Do not add this tail again if the max-plus scheduler already includes scalar completion.

If address/masking instructions and tensor dispatch share an issue resource d, do not assume independent overlap:

\[
C_d=\left\lceil(n_{addr}+n_{mask}+n_{decode,d}+n_{tensor\_dispatch})/I_d\right\rceil,
\quad T_{iter}^{LB}\ge\max(T_{compute},C_d,C_{other}),
\]
\[
T_{stall,shared}^{LB}=\max(0,C_d-T_{compute}).
\]

Dependency blocking can increase this bound; only the schedule determines the modeled exact penalty. Example: compute=100, scalar=60 produces zero exposed tail on distinct engines; if tensor dispatch=70 and scalar demand=60 on one slot/cycle, the shared demand is 130 and the bound adds 30. Producers and consumers in different warps can still contend on a shared dispatcher. On Ascend, model scalar issue, Vector, Cube, MTE1/2/3 and their specific control dependencies rather than importing CUDA issue topology.

### 3.3 Rings, steady-state overlap, and ragged experts

For each versioned slot, schedule tasks v with precedences and resource availability as in §3.1. Copy completion must precede decode; decode publication precedes contraction; operand-read completion precedes overwrite. A producer stall waiting for a free slot and a consumer stall waiting for ready data are explicit wait nodes.

For identical independent stages i, sufficient buffering, and no shared resource, ideal period P=max_i T_i. A useful necessary ring constraint is D≥⌈H/P⌉ where H is slot holding time from reservation to release; actual feasibility still requires schedule simulation. For J tiles, ideal T_total≈Σ_i T_i+(J−1)P plus launch, routing, and drain costs. Path A can have a decode/dispatch dependence that forbids this full overlap. Path C can hide it only when resource capacities and slot ownership allow it.

Ragged expert work must be summed over per-expert M_e, including zero experts and padded contraction dimensions:

\[
F_{useful}=2NK\sum_e M_e,\quad
F_{exec}=\sum_e 2\,pad(M_e,m_0)\,pad(N,n_0)\,pad(K,k_0).
\]

Weight input is NK(1/2+1/B) bytes per active expert for one-byte local scales, assuming one fetch; actual traffic uses the resident-loop dependency model and cache policy. Repeated M tiles may decode or reload weights repeatedly. For N=K=4096:

| B | Packed weights + block scales, MiB | Weight-only intensity, FLOP/byte |
|---|---:|---:|
| 16 | 9.00 | 3.5556 M |
| 32 | 8.50 | 3.7647 M |
| 64 | 8.25 | 3.8788 M |

The tensor multiplier and expert descriptors add metadata. Intensity is a bound excluding activations, outputs, cache misses, and routing. Full FP8 materialization adds NK bytes of writes and NK bytes of reads (32 MiB here), besides packed reads; fusion removes those global intermediates but still pays local FP8 traffic. At M=1/8, contraction padding, launch and weight-fetch costs matter; at M=2048, contraction/reuse generally matter more. This does not establish a throughput ranking without calibrated service times.

### 3.4 Backend change contract

Inspection of the sibling PipeSim checkout identifies `topology.py:DType.size_bytes` as integer-byte based, `dsl.py` tensor `nbytes`/strides as dtype-size based, and `evaluator.py` as a three-tier feasibility/roofline/max-plus evaluator. Its existing profiles describe analytical Ascend resources. Implement packed bit-width as an explicit storage property (separate from arithmetic dtype), exact byte rounding, intermediate buffer lifetimes, scalar/dispatch resources, per-op instruction templates, and bank transaction demand before evaluation. Do not reinterpret existing generic FP8 Cube support as proof of native E4M3/E5M2 format support on every SoC.

Add profile fields: bank count/width/ports/access granularity/hash, instruction lane widths and issue capacities, latency intervals, shared-dispatch incidence, scale/convert support, copy burst granularity, ring synchronization primitives, and contraction operand requirements. Tag each coefficient with device/tool version, experiment ID, provenance, confidence interval, and calibration date. Preserve existing FP16-only flows through opt-in packed stages.

## 4. Memory layout and bank audit: `AKA3006` / `AKA3007`

### 4.1 Byte and bit geometry

For row r, column c, payload row pitch P4 bytes:

\[
a_4(r,c)=base_4+rP_4+\lfloor c/2\rfloor,\quad
shift(c)=4(c\bmod2).
\]

For unpacked FP8 row pitch P8 bytes:

\[
a_8(r,c)=base_8+rP_8+c,\quad P_4\ge\lceil K_t/2\rceil,\quad P_8\ge K_t.
\]

P8 is not necessarily 2P4 after padding. Analyze input loads, expanded stores, operand transfer loads, and scale accesses independently. Require disjoint source/destination spans unless in-place expansion has a proven overwrite-safe ordering; ring slots and swizzled extents count against capacity.

For a linear B_bank-bank memory with bank word width w bytes:

\[
bank(a)=\lfloor a/w\rfloor\bmod B_{bank}.
\]

For every instruction/transaction, enumerate the actual active-lane addresses and each bank word touched by its vector width. For bank j, let U_j be the number of distinct service units required after permitted read broadcast/coalescing. With p_j ports:

\[
C_{bank}=\max_j\lceil U_j/p_j\rceil.
\]

Evaluate requests within the hardware-defined transaction groups, not one aggregate over an entire warp/vector repeat. Consecutive FP8 bytes sharing a 4-byte NVIDIA bank word are not four conflicting full-word operations by definition; store combination, overlapping writes and transaction splitting must be modeled. Same-word read broadcast is permitted only where documented, never assumed for writes. `AKA3006` emits a concrete instruction, lane/address witness, bank loads, service multiplier, and remediation. Unknown transaction rules mean unproven, not a fabricated exact multiplier.

### 4.2 Orthogonality proof and required padding

For a **column-wise** group of B_bank lanes, one bank word per lane, fixed column byte offset, no swizzle, and word-aligned base/pitch, set d=P8/w. Then:

\[
bank_\ell=(bank_0+\ell d)\bmod B_{bank}.
\]

Distinctness is equivalent to gcd(d,B_bank)=1: a collision requires (ℓ₁−ℓ₂)d≡0 mod B_bank; the smallest positive colliding separation is B_bank/gcd(d,B_bank). Thus B_bank/gcd(d,B_bank) banks are visited and gcd(d,B_bank) lanes share each bank for a full-bank lane group. For any power-of-two bank count (8 or 32), d must be odd. If P8/w is even, padding by one bank word makes it odd; if already odd, no padding is required. This proves sufficiency and minimal **word-granularity** padding under these assumptions, not universal byte padding.

| Geometry (illustrative except NVIDIA linear bank width) | Unpadded P8 | d | Distinct banks per B_bank lanes | Padding | Padded d |
|---|---:|---:|---:|---:|---:|
| 32 banks, w=4 bytes | 128 | 32 | 1 | +4 bytes | 33, all 32 banks |
| 8 banks, w=32 bytes, profile assumption | 256 | 8 | 1 | +32 bytes | 9, all 8 banks |
| 32 banks, w=4 bytes | 132 | 33 | 32 | 0 | 33 |

The existing PipeSim UB model uses an 8-bank/32-byte geometry assumption; calibrate it per SoC rather than claiming all NPUs use it. With 32 lanes and 8 single-ported banks, ≥4 service rounds are unavoidable for 32 distinct word requests even with an orthogonal pitch. Orthogonal means no avoidable collision beyond the bank-count limit, not one-cycle service.

Byte-strided pitches not divisible by w require enumeration with floor(a/w); the gcd shortcut is invalid. Hashed/swizzled layouts require the profile bank function, transaction splitter, and accessed words. NVIDIA [shared-memory banking](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks) supplies the 32-bank parallel; the NPU bank map remains a profile input.

**Descriptor compatibility constraint:** if P8 must also be a multiple of A_desc with A_desc/w even, every permitted word stride is even. No simple row padding can satisfy both descriptor alignment and full column-bank orthogonality. Example: 16-byte pitch alignment with w=4 forces d to be a multiple of 4. Use a supported XOR/swizzled/fractal layout or change the lane access pattern, then prove the new mapping. Do not recommend +4 padding if it invalidates the contraction descriptor. Verify base, leading/outer strides, element format, shape, swizzle, descriptor bit-field range, and allocation extent using the selected contraction instruction contract; reuse existing base/stride/domain diagnostics for descriptor violations.

### 4.3 Diagnostic division

`AKA3006` remains the existing **observed or provable per-operation bank contention** diagnostic. `AKA3007` identifies the **structural non-orthogonal sub-byte expansion stride** when the column-word-access preconditions above hold and gcd(d,B_bank)>1, or a swizzled mapping has a proven avoidable collision. It can fire before a concrete consumer instruction exists. Link to and deduplicate `AKA3006` for the same layout/root cause. Do not trigger `AKA3007` solely because a normal row-contiguous byte stream has an even pitch. Unknown affine/symbolic lanes report existing analyzability diagnostics with missing proof context.

The inspected analyzer currently narrows `AKA3006` to dual-source Vector UB base-offset collisions using an 8-bank assumption. Preserve that check while extending it to actual touched words/transaction groups. For two bases a and b, compare floor(a/w) mod B_bank and floor(b/w) mod B_bank; floor(|a−b|/w) mod B_bank is equivalent only with suitable alignment assumptions. For example w=32, a=31 and b=32 occupy different banks although floor(|a−b|/w)=0. Base separation by one bank word proves orthogonality only for equal relative access patterns whose bank maps remain distinct; vector strides and multiple touched words require the full audit.

## 5. Synchronization, allocation, and diagnostic contracts

### 5.1 Versioned happens-before invariants

Represent each staging object as (domain, byte interval, slot i, generation g), with ownership and participant scope. Required state machine:

```text
FREE(g) → COPY_PENDING(g) → PACKED_READY(g) → DECODE_WRITING(g)
        → FP8_READY(g) → OPERAND_READ_IN_FLIGHT(g) → FREE(g+1)
```

Required edges for all intersecting source/destination byte ranges:

1. Reserve FREE(i,g) before any copy/write to that slot; initialize the barrier/event and expected participant/byte counts before completion can be reported.
2. Every packed-byte and scale-copy completion happens-before the decoder's corresponding read. An empty/masked tile completes a defined empty-stage path; it must not leave consumers waiting on nonexistent transactions.
3. Every decoder write happens-before publication of FP8_READY(i,g), which happens-before every operand read. Compiler/thread ordering and inter-actor synchronization must both establish this edge.
4. Every asynchronous contraction operand-read completion happens-before RELEASE(i,g), which happens-before overwrite/reuse for generation g+1. Issue of MMA alone is not read completion. Conservative release at contraction completion is legal.
5. Producer and consumer wait on the same barrier/event identity, slot generation/phase, domain, and scope; all required participants arrive on every feasible control path. No early exit may strand the ring.
6. Cross-CTA distributed memory requires cluster-scoped accessibility and ordering, residency/lifetime guarantees, and a supported transfer mechanism; CTA-local barriers do not establish those edges.

For CUDA TMA, expected transaction bytes correspond to the asynchronous copy payload, not the number of FP8 bytes eventually decoded. Raw-copy completion publishes **packed** bytes; it does not publish later generic shared-memory decode stores. Those stores require producer/consumer ordering and, when crossing generic/async proxies, the applicable proxy fence before WGMMA/other async use. Follow the exact [PTX async proxy, mbarrier, WGMMA and tensor-copy semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/). Do not manually count scalar decode stores as TMA transaction completions. Avoid double counting `arrive.expect_tx` combined operations.

For Ascend, construct equivalent edges from the actual MTE/Vector/Cube event set/wait pairs, queue enqueue/dequeue ownership, and operand-transfer completion. A high-level `DeQue` establishes only the queue contract it documents; verify event endpoints and domain transitions in the profile. TMA `mbarrier` terminology does not prescribe an NPU instruction sequence.

Use control-flow/dominance analysis for publication, interval aliasing for data hazards, and loop-carried slot-generation analysis for reuse. A memory read is safe iff every feasible reaching overlapping write is ordered, or the actor owns a provably initialized immutable buffer. Build the wait/resource graph to detect cycles independently of per-buffer races. Dynamic sizes need a witness or a conservative unproven result.

### 5.2 Live SRAM, fragmentation, and register pressure

For pool p at time t, with aligned allocation sizes S_i, padding, ring depth D_i, descriptor/scales/temporary storage included:

\[
Live_p(t)=\sum_{i\ live\ at\ t} D_i\,align(S_i,A_i),\qquad
Peak_p=\max_t Live_p(t).
\]

Use actual padded row extents, not just useful tile area. If each generation is independently represented, do not multiply by D_i again. Domains such as UB/L1/L0A/L0B/L0C cannot be pooled merely because their total free bytes suffice. For fixed placement, check all allocation intervals against capacity and each other through overlapping lifetimes. For dynamic placement, aggregate free bytes F≥request S is insufficient; require a suitably aligned contiguous free interval. Compute the largest aligned usable hole H_A. Fragmentation witness: F≥S but H_A<S. Capacity witness: Peak_p>capacity_p. A proven impossible allocation is not automatically an actual GM spill; report the required fallback and extra traffic only if the program/backend specifies that fallback.

Registers are a separate allocation domain. Live ranges of packed words, expanded words, scalar scales, RNG, pointers, and accumulators determine pressure. NVIDIA's per-thread limit of 255 registers is not a full-occupancy target. Resident blocks obey the minimum of register, shared-memory, warp, thread, block and any cluster limits after allocation rounding. Parameterize register allocation granularity; use compiled allocation/spill counts when available. Existing performance/analyzability diagnostics can report register risk without conflating it with `AKA1012` SRAM capacity. [Hopper occupancy limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) support this distinction; use a separate SM120 profile locally.

### 5.3 Exactly three proposed new rules

The current sibling analyzer's `diagnostics.py` assigns memory codes through AKA1011, synchronization codes through AKA2010, and performance codes through AKA3006. The requested next codes are unused in that inspected checkout.

| Code and enum proposal | Trigger / proof | Severity | Required witness and remediation |
|---|---|---|---|
| **AKA2011 `UNSYNCHRONIZED_DEQUANT_STAGE`** | Packed or FP8 staging read lacks a reaching-write HB edge; overwritten slot still has outstanding operand readers; barrier/event phase, scope or transaction expectation fails publication | FATAL for proven race/illegal wait; WARNING for unresolved ordering | Producer/consumer locations, pool/range, slot/generation, missing edge/event endpoints; add correct publication wait/fence or delay release until read completion |
| **AKA1012 `FUSED_UNPACK_SRAM_ALLOCATION`** | Padded/ring-buffered fused unpack allocation exceeds per-pool capacity, or sufficient aggregate free bytes cannot fit the required aligned contiguous extent | FATAL for proven allocation failure without legal fallback; WARNING for explicit spill fallback or unresolved dynamic peak | Pool, capacity, peak live bytes, lifetimes, aligned request, holes; reduce tile/ring depth, reuse nonoverlapping lifetimes, or use a declared fallback |
| **AKA3007 `SUBBYTE_NONORTHOGONAL_STRIDE`** | Proven expansion/consumer geometry meets §4.2 preconditions and gcd(d,B_bank)>1, or a supported bank map proves avoidable collisions | WARNING | P4/P8, bank count/width, active lanes, gcd or bank witness, transaction width and descriptor restrictions; pad by a legal bank word, swizzle, or change the access map |

`AKA1001` continues to cover generic capacity overflow; `AKA1012` adds the fused-stage lifetime/fragmentation witness and deduplicates a generic same-root failure. Existing `AKA3004` remains a general fragmentation advisory; `AKA1012` identifies inability to place a required fused-stage allocation. Do not renumber `AKA3006`. Every finding includes source span, concrete repair, proof status, hardware profile, and the originating conversion stage.

### 5.4 Proposed pass order and backend touchpoints

1. **Normalize format/storage:** payload nibble order, scale groups, bit widths, dimensions, ragged predicates and padded execution shape. Preserve original source locations.
2. **Check scale rewrite legality:** distinguish k-invariant factors from reduction-dependent factors; require range/rounding guards for exponent shortcuts.
3. **Infer stage ranges and lifetimes:** packed/scales/FP8/accumulators/descriptors; resource-domain compatibility and exact padded footprints.
4. **Audit placement and descriptors:** extend `checkers/memory.py` allocation checks and symbolic solver bounds; generic alignment errors keep existing IDs; fused placement uses AKA1012.
5. **Audit transaction banks:** extend `hardware.py` geometry plus `checkers/memory.py` lane/word mapping; AKA3006 and AKA3007 share witnesses.
6. **Verify publication/reuse:** extend synchronization IR/checkers with actor and slot-generation edges; AKA2011 with the missing edge or a cycle witness.
7. **Lower costs and schedule:** PipeSim `dsl.py` carries stages/storage bits; `topology.py` carries capabilities/resources; `evaluator.py` computes packed traffic and resource-constrained schedules before roofline/latency reports.

Semantic rewrites and unsupported FP8 contraction formats must be rejected or reported through existing domain/unsupported-intrinsic diagnostics. The three new codes are not substitutes for a format-capability checker.

## 6. Calibration protocol and acceptance cases

### 6.1 Calibration datasets

Collect separate measurements for raw register embedding, shared/UB expansion, E4M3 scale decode/multiply/RNE conversion, power-of-two exponent shortcut, producer publication, and complete contraction. Sweep elements, lane participation, register live depth, vector repeat counts, stride, scale B and ring depth. Linear fits of time versus G/S/bytes identify service coefficients; dependent instruction chains identify latency. Independent chains identify issue throughput. Use the scalar-versus-contraction matrix to distinguish exposed tail from shared dispatch.

For GPU profiles collect NCU metrics using the device's queried metric names: elapsed SM cycles; integer/FP/LSU/Tensor instruction counts; issued/eligible/active warps; shared transactions and bank conflicts; register allocation; local-memory spill bytes; long/short-scoreboard, math-pipe-throttle, barrier and wait samples. Stall samples are sampled states, not additive exact cycle partitions. Replay can change cache behavior; label warm/cold policy, concurrent load, clocks/power, iterations and dispersion. See [Nsight Compute profiling methodology](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

NPU calibration needs corresponding Vector/Cube/MTE/scalar service experiments on the target SoC; GPU coefficients must not be copied into an Ascend profile. Existing PipeSim manifest constants remain analytical priors until such measurements exist. Report unknown unsupported counters explicitly.

### 6.2 Minimum deterministic acceptance matrix

| Case | Required outcome |
|---|---|
| All 16 E2M1 codes to E4M3 and E5M2 | Exact value and sign, including −0 |
| Every nonnegative finite E4M3 scale × raw E2M1 magnitude | Reference RNE matches neighboring-value/tie-even oracle; some products necessarily rounded |
| Power-of-two scale moving into FP8 subnormal or overflow | Guarded shift/round/saturate; reject unchecked byte exponent add |
| Scale depends on k, moved to alpha | Rewrite rejected; two-vector counterexample preserved |
| N=K=4096, B=16 | 8 MiB packed payload + 1 MiB local scales, no global FP16 intermediate |
| 4096 values / 32 lanes / 8 nibbles per word | G=16; native conversion operation counts remain separately represented |
| compute=100, scalar=60, independent resources | exposed-tail stall=0 |
| tensor-dispatch=70, scalar=60, one shared slot/cycle | resource lower bound=130, extra bound=30 |
| 32-bank w=4, P8=128, column-word access | AKA3007; 32-way single-bank requests; +4 yields orthogonal map if descriptor permits |
| 8-bank w=32, P8=256, 8-lane word access | AKA3007; +32 yields all 8 banks |
| Same pitches, contiguous row-byte access | No structural stride warning absent an actual conflict witness |
| 16-byte pitch alignment, 32-bank w=4 column map | +4 rejected by descriptor; require swizzle/access remap |
| Copy barrier complete but decode publication absent | AKA2011 on FP8 consumer read |
| Consumer issued asynchronous contraction, producer overwrites | AKA2011 until operand-read completion/release |
| F=128 bytes, request=96, usable aligned holes=64+64 | AKA1012 fragmentation despite sufficient aggregate free bytes |
| Two live rings fit individually but overlap above UB capacity | AKA1012 with combined lifetime witness |
| Disjoint lifetimes use identical interval | No capacity/race diagnostic |
| M=0 expert or masked tail skips copy | Defined empty-stage completion; no unbalanced barrier |

Acceptance of a performance coefficient requires independent holdout sizes, versioned raw measurements, and a declared tolerated error range. Acceptance of a safety pass requires both a positive witness case and a corrected case, including symbolic dimensions/control flow. Do not claim an optimal path until a complete fused schedule is measured against the materialized and native-block-scaled controls.

## 7. Numerical metrics and interpretation

Measure weight and output relative Frobenius error ||X−Xhat||F/||X||F and flattened cosine. Define zero-reference cases explicitly in production. KL is measured only on distributions: here per-token softmax over synthetic projection outputs with temperature 1, KL(p_ref||p_candidate) averaged across tokens. It is not a KL between signed weight arrays or evidence of language-model perplexity. Each test uses FP16-rounded original weights/activations and an FP32-accumulated reference. This isolates storage/operand drift; it is not bit-identical to every FP16 Tensor Core accumulator mode.

The study compares original FP16-rounded weights, direct tensor-scaled FP8, reconstructed FP4 without an FP8 cast, and reconstructed FP4→E4M3/E5M2 with RNE/stochastic rounding. FP8 variants use tensor-scaled E4M3 activations as well; the reconstructed-FP4 control retains baseline activations. E4M3 block scales use a tensor factor to fit the finite local-scale range; E8M0 uses upward-rounded power-of-two block factors. Keep those policy choices in result metadata. Generated Gaussian weights have variance 1/K; the heavy-tail mixture multiplies 0.2% of weights by 32 and 0.5% of activations by 16. Seed is 20261004, synthetic activation M=32. These are controlled outliers, not traces from DeepSeek experts.

For stricter ablation add actual gate/up/down tensors, activation-only and weight-only casts, routed expert histograms, multiple seeds, underflow/clip counters, FTZ/canonical-zero variants, and downstream gate/nonlinearity effects. The exhaustive raw-code and scale-product checks establish encoding behavior; GPU hardware subnormal behavior needs a separate probe rather than inference from software decoding.

## 8. Reproducible experiments and measured results

Run from the research workspace:

```powershell
python research_nvfp4_fp8/run_study.py --full
python research_nvfp4_fp8/verify_and_report.py
```

Dependencies: NumPy, CUDA-enabled PyTorch. Output is `research_nvfp4_fp8/results/study.json`; tables below are derived from that file. Actual `_scaled_mm` controls use pre-materialized E4M3 operands, FP32 output, `use_fast_accum=False`, 5 warmups and five batches of 20 repetitions timed by CUDA events. M=1/8 are physically padded to 16; the report keeps both requested and executed M. Host routing, materialization, and decode latency are excluded. Working-set/cache behavior is whatever repeated cuBLAS calls achieve; these are warm repeated-operation controls, not verified cold DRAM throughput. Timings must not calibrate conversion coefficients.

<!-- MEASURED_RESULTS -->

### 8.1 Encoding and software fidelity results

Exhaustive software check: all 16 signed raw codes embed exactly; **275/1,016** nonnegative finite E4M3-scale products need E4M3 rounding (including overflow saturation). Both E4M3/E5M2 mappings and CUDA conversion behavior are independently checked by `verify_and_report.py`. CUDA casts preserve all finite representable values tested, including subnormals and −0; this does not establish Tensor Core operand subnormal behavior.

B=16, E4M3 local scales; errors below are relative to the original FP16-rounded baseline. FP8 rows include activation quantization.

| Distribution | Weight path / rounding | Weight relative error | Output relative error | Output cosine | Softmax KL |
|---|---|---:|---:|---:|---:|
| gaussian | direct_fp8 / rne | 0.026489 | 0.037619 | 0.999292 | 0.000692822 |
| gaussian | nvfp4_reconstructed / none | 0.095102 | 0.095895 | 0.995401 | 0.00454214 |
| gaussian | nvfp4_to_e4m3 / rne | 0.095622 | 0.099773 | 0.995027 | 0.00489804 |
| gaussian | nvfp4_to_e4m3 / stochastic | 0.097163 | 0.101234 | 0.994873 | 0.00505245 |
| gaussian | nvfp4_to_e5m2 / rne | 0.110476 | 0.114145 | 0.993478 | 0.00643873 |
| gaussian | nvfp4_to_e5m2 / stochastic | 0.115483 | 0.119109 | 0.992920 | 0.00705181 |
| heavy_tail | direct_fp8 / rne | 0.026415 | 0.037488 | 0.999297 | 0.0240844 |
| heavy_tail | nvfp4_reconstructed / none | 0.083931 | 0.084126 | 0.996461 | 0.0105375 |
| heavy_tail | nvfp4_to_e4m3 / rne | 0.084228 | 0.088497 | 0.996082 | 0.0232599 |
| heavy_tail | nvfp4_to_e4m3 / stochastic | 0.084819 | 0.089040 | 0.996032 | 0.0230636 |
| heavy_tail | nvfp4_to_e5m2 / rne | 0.106676 | 0.108874 | 0.994401 | 0.0783193 |
| heavy_tail | nvfp4_to_e5m2 / stochastic | 0.105996 | 0.108806 | 0.994078 | 0.124862 |

All 62 cases, including B=32/64 and E8M0 scale ablations, remain in the JSON. In this seed the B=16 E4M3/RNE reconstructed route has output error 9.98% (Gaussian) and 8.85% (heavy-tail), while direct FP8 has 3.76% and 3.75%. The heavy-tail softmax KL does not track Frobenius error monotonically. No quality claim extends to trained expert layers.

### 8.2 Actual GPU FP8 GEMM controls

| Requested M | Executed M | B | Median µs | Batch min–max µs |
|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 15.757 | 15.034–19.557 |
| 8 | 16 | 16 | 22.555 | 14.998–29.560 |
| 32 | 32 | 16 | 15.170 | 15.122–16.749 |
| 128 | 128 | 16 | 39.698 | 39.640–39.795 |
| 2048 | 2048 | 16 | 538.381 | 512.763–558.491 |
| 1 | 16 | 32 | 15.059 | 14.986–15.694 |
| 8 | 16 | 32 | 15.168 | 14.994–25.874 |
| 32 | 32 | 32 | 15.179 | 15.170–15.211 |
| 128 | 128 | 32 | 39.773 | 39.675–48.874 |
| 2048 | 2048 | 32 | 530.130 | 511.966–551.662 |
| 1 | 16 | 64 | 15.778 | 15.171–26.891 |
| 8 | 16 | 64 | 15.347 | 15.088–28.200 |
| 32 | 32 | 64 | 15.226 | 15.224–15.342 |
| 128 | 128 | 64 | 39.771 | 39.754–39.830 |
| 2048 | 2048 | 64 | 515.243 | 512.434–533.117 |

The control GEMM no longer reads FP4 block scales, so timing differences between B values reflect run variation/data/library effects, not measured decode performance. Requested M=1 and M=8 execute the same padded shape. The maximum output drift from FP32 matmul on the same prequantized operands was 2.73e-09 in this run; this checks the control, not full quantization fidelity.
<!-- END_MEASURED_RESULTS -->

Unmeasured in this execution: complete fused A/C/D latency, NCU per-pipeline issue/stall decomposition, GPU register/spill counts for the conceptual mapping, Hopper/datacenter Blackwell paths, and NPU service coefficients. Those remain analytical/profile-dependent; no exact hardware stall numbers are inferred from the synthetic fidelity results.

## 9. Delivery boundary

This document specifies the backend updates; it does not modify sibling analyzer or PipeSim repositories. The small companion experiment measures numeric transformations and FP8 GEMM controls. The only FP4→FP8 implementation illustration is the 32-line conceptual extraction snippet in §2.2. Profile coefficients must be calibrated on each intended backend before the proposed model can predict absolute fused-kernel latency.
