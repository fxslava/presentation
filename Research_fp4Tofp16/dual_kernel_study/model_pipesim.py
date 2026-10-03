#!/usr/bin/env python3
"""
model_pipesim.py -- PipeSim spatial-temporal modeling for the two FP4->FP16
dequantization kernel designs (Ascend 910B / "950PR" label, DaVinci v2 model).

Benchmark problem (identical for both kernels):
  Dequantize W16 [2048 x 4096] fp16 (16 MiB) from NVFP4 storage:
    - packed FP4 weights (E1M2 grid per repo default), 2 nibbles/byte -> 4 MiB
    - E4M3 block scales, 1 B / 16 weights                       -> 512 KiB
  => 8,388,608 output elements, 524,288 scale groups, 32,768 16x16x32 tiles.

Modeling notes
  * PipeSim has no 0.5 B dtype; FP4 operands are modeled as int8 (1 B/elem),
    the closest sub-byte proxy.  Raw MTE2 A/B bytes therefore OVERESTIMATE the
    real fp4 traffic by exactly 2x -- corrected numbers are emitted alongside
    ("corr" columns) using bytes_fp4 = bytes_model / 2.
  * Kernel A vector op chain (int8-lane nibble split -> SM->TC sign repair ->
    Cast int8->half -> per-block scale Mul): 7.4 ops/element (hand-counted,
    incl. amortized scale decode/broadcast).
  * Kernel B AIV epilogue (repo stage): Cast + per-group Duplicate offset tile
    (16/256) + integer exp Add + zero-guard chain (And,Min,Sub,And): 6.06
    ops/element on fp16 lanes.
"""
import json
import sys

sys.path.insert(0, r"D:\Projects\PipeSim\src")

from ascend_pipesim import AlgorithmFlow, BufferPlan, DType, evaluate

ARCH = "ascend910b"
ELEMENTS = 2048 * 4096          # 8,388,608 fp16 outputs
TILES = ELEMENTS // 256         # 32,768 tiles of [16 groups x 16 weights]

OUT_DIR = r"D:\Projects\Presentation\Research_fp4Tofp16\dual_kernel_study"


def stages_of(rep):
    return {s.name: round(s.cycles_per_step, 2) for s in rep.schedule.stages}


def rep_json(name, rep, extra=None):
    d = {
        "name": name,
        "valid": rep.is_valid,
        "summary": rep.summary,
        "limiting_stage": rep.schedule.limiting_stage,
        "secondary": rep.schedule.secondary_stage,
        "slack_pct": round(rep.schedule.slack_pct, 1),
        "stages_cyc_per_step": stages_of(rep),
        "steps": rep.schedule.steps,
        "total_cycles": round(rep.schedule.total_cycles),
        "t_min_cycles": round(rep.t_min_cycles),
        "t_max_cycles": round(rep.t_max_cycles),
        "total_us": round(rep.timing_us.get("total_us", 0.0), 3),
        "traffic": {k: round(v) for k, v in rep.traffic.items()},
        "violations": [v.code + ": " + v.message for v in rep.violations],
        "footprints": {pool: {"bytes": round(fp.bytes_used),
                              "pct": round(100.0 * fp.bytes_used / fp.capacity_bytes, 1),
                              "breakdown": fp.breakdown}
                       for pool, fp in rep.sram_footprints.items()},
    }
    if extra:
        d.update(extra)
    return d


results = {"arch": ARCH, "elements": ELEMENTS, "tiles": TILES}

# ---------------------------------------------------------------------------
# KERNEL A -- naive AIV vector dequant, GM -> UB -> Vector -> UB -> GM
#   modeled as ELEMWISE; dtype uint8 is the sub-byte proxy for packed FP4.
#   n_inputs=1 folds the 1/16 B/elem scale stream into the notes (0.0625 B/el).
# ---------------------------------------------------------------------------
OPS_PER_ELEM_A = 7.4
best_a = None
sweep_a = []
for tile_elems in (4096, 8192, 16384, 24576):
    flow = AlgorithmFlow.elementwise(
        elements=ELEMENTS, tile_elements=tile_elems,
        dtype="uint8", out_dtype="fp16", n_inputs=1,
        ops_per_element=OPS_PER_ELEM_A, name=f"aiv_naive_t{tile_elems}",
    )
    rep = evaluate(flow, ARCH)
    entry = rep_json(f"kernelA_tile{tile_elems}", rep)
    # sub-byte correction: in_bytes real = 0.5625 B/el (0.5 packed + 1/16 scale)
    mte2_c = entry["stages_cyc_per_step"].get("MTE2", 0.0)
    entry["MTE2_cyc_per_step_corrected"] = round(
        mte2_c * (0.5625 / 1.0), 2)  # uint8 model says 1.0 B/el
    sweep_a.append(entry)
    if rep.is_valid and (best_a is None or
                         rep.schedule.total_cycles < best_a.schedule.total_cycles):
        best_a = rep
results["kernelA_sweep"] = sweep_a
results["kernelA_best"] = rep_json("kernelA_best", best_a)

# ---------------------------------------------------------------------------
# KERNEL B -- Cube trick, pseudo-GEMM per tile: m=16P, n=16, k=32P (fp4),
#   batch = TILES / P panels.  int8 proxy dtype; A/B MTE2 bytes corrected 2x.
#   subtile = full fractal extent (one LoadData2D per operand per tile, which
#   is what the hardware does -- mStep/kStep extend over the whole panel).
# ---------------------------------------------------------------------------
sweep_b = []
for P in (1, 2, 4, 8):
    m, n, k = 16 * P, 16, 32 * P
    batch = TILES // P
    flow = AlgorithmFlow.gemm(
        m=m, n=n, k=k, batch=batch,
        tile=(m, n, k), subtile=(m, n, k),
        dtype="int8", out_dtype="fp32",
        buffers=BufferPlan.double(),
        output_path="fixpipe_gm", name=f"cube_trick_P{P}",
    )
    rep = evaluate(flow, ARCH)
    entry = rep_json(f"kernelB_P{P}_fp32stage", rep)
    st = entry["stages_cyc_per_step"]
    entry["MTE2_cyc_per_step_corrected"] = round(st.get("MTE2", 0.0) / 2.0, 2)
    entry["cycles_per_256_outputs"] = round(rep.schedule.total_cycles / ELEMENTS * 256, 3)
    sweep_b.append(entry)
results["kernelB_sweep"] = sweep_b

# Variant: Fixpipe converts to fp16 inline (halves staging traffic and removes
# the AIV Cast op: 5.06 ops/elem on a 2 B read stream).
flow = AlgorithmFlow.gemm(
    m=16, n=16, k=32, batch=TILES,
    tile=(16, 16, 32), subtile=(16, 16, 32),
    dtype="int8", out_dtype="fp16",
    buffers=BufferPlan.double(),
    output_path="fixpipe_gm", name="cube_trick_P1_fp16stage",
)
rep = evaluate(flow, ARCH)
results["kernelB_fp16stage"] = rep_json("kernelB_P1_fp16stage", rep)

OPS_PER_ELEM_BV16 = 5.06
flow = AlgorithmFlow.elementwise(
    elements=ELEMENTS, tile_elements=8192,
    dtype="fp16", out_dtype="fp16", n_inputs=1,
    ops_per_element=OPS_PER_ELEM_BV16, name="cube_trick_aiv_epilogue_fp16",
)
rep = evaluate(flow, ARCH)
results["kernelB_aiv_epilogue_fp16"] = rep_json("kernelB_aiv_epilogue_fp16", rep)

# ---------------------------------------------------------------------------
# KERNEL B -- AIV epilogue pass: gmC fp32 -> cast fp16 -> int exp add ->
#   zero guard -> gmOut fp16 (reads gmS scales as well, 1/16 B/el)
# ---------------------------------------------------------------------------
OPS_PER_ELEM_BV = 6.06
flow = AlgorithmFlow.elementwise(
    elements=ELEMENTS, tile_elements=8192,
    dtype="fp32", out_dtype="fp16", n_inputs=1,
    ops_per_element=OPS_PER_ELEM_BV, name="cube_trick_aiv_epilogue",
)
rep = evaluate(flow, ARCH)
results["kernelB_aiv_epilogue"] = rep_json("kernelB_aiv_epilogue", rep)

# ---------------------------------------------------------------------------
# Roofline-style single-core summary at the achieved traffic (analytical)
# ---------------------------------------------------------------------------
def per_elem_cycles(entry, corr_key=None):
    return {k: round(v / ELEMENTS, 6) for k, v in entry["stages_cyc_per_step"].items()}

results["per_element_cycles"] = {
    "kernelA_best": per_elem_cycles(results["kernelA_best"]),
    "kernelB_P1": per_elem_cycles(sweep_b[0]),
    "kernelB_aiv_epilogue": per_elem_cycles(results["kernelB_aiv_epilogue"]),
}

# ---------------------------------------------------------------------------
# Vector ISSUE-overhead adjustment (not priced by the elementwise model).
#
# Both designs must materialize a per-16-element (per NVFP4 block) lane
# broadcast on the vector pipe:
#   Kernel A (pair-planar): a 16-elem block spans 8 even + 8 odd nibble
#     lanes, BUT the scale VALUE is identical for both planes' runs -- one
#     shared 4096-lane scale vector (runs of 8) serves both plane passes
#     -> 1 Duplicate issue / 16 elems, same rate as Kernel B.
#   Kernel B (sequential):  exponent-offset tile rows are 16 lanes wide
#     -> 1 Duplicate issue / 16 elems (repo AIV stage structure).
# PipeSim prices vector WORK bytes but not instruction issue; each small
# Duplicate costs ~arch.vector.issue_overhead_cycles (4) on the same pipe.
# ---------------------------------------------------------------------------
ISS_CYC = 4.0
adj = {
    "issue_overhead_cycles_per_duplicate": ISS_CYC,
    "kernelA": {
        "dup_issues_per_element": 1.0 / 16.0,
        "issue_cyc_per_element": round(ISS_CYC / 16.0, 4),
        "vec_work_cyc_per_element": round(
            results["kernelA_best"]["stages_cyc_per_step"]["VECTOR"] / ELEMENTS, 6),
    },
    "kernelB_aiv_epilogue_fp32": {
        "dup_issues_per_element": 1.0 / 16.0,
        "issue_cyc_per_element": round(ISS_CYC / 16.0, 4),
        "vec_work_cyc_per_element": round(
            results["kernelB_aiv_epilogue"]["stages_cyc_per_step"]["VECTOR"] / ELEMENTS, 6),
    },
    "kernelB_aiv_epilogue_fp16": {
        "dup_issues_per_element": 1.0 / 16.0,
        "issue_cyc_per_element": round(ISS_CYC / 16.0, 4),
        "vec_work_cyc_per_element": round(
            results["kernelB_aiv_epilogue_fp16"]["stages_cyc_per_step"]["VECTOR"] / ELEMENTS, 6),
    },
}
TILE_A = 8192    # kernel A implementation tile (elements per MTE2 step)
TILE_BV = 8192   # kernel B AIV-epilogue tile (elements per MTE2 step)
adj["kernelA"]["vec_work_cyc_per_element"] = round(
    [e for e in sweep_a if e["name"] == "kernelA_tile8192"][0]
    ["stages_cyc_per_step"]["VECTOR"] / TILE_A, 6)
adj["kernelB_aiv_epilogue_fp32"]["vec_work_cyc_per_element"] = round(
    results["kernelB_aiv_epilogue"]["stages_cyc_per_step"]["VECTOR"] / TILE_BV, 6)
adj["kernelB_aiv_epilogue_fp16"]["vec_work_cyc_per_element"] = round(
    results["kernelB_aiv_epilogue_fp16"]["stages_cyc_per_step"]["VECTOR"] / TILE_BV, 6)
for key, v in adj.items():
    if isinstance(v, dict) and "issue_cyc_per_element" in v:
        v["vec_total_cyc_per_element"] = round(
            v["issue_cyc_per_element"] + v["vec_work_cyc_per_element"], 6)
# composed single-pair wall clocks (cross-core producer/consumer, double
# buffered): wall = ramp + max(AIC per-tile, AIV per-tile) * tiles.
# AIC per tile at real fp4 density: MTE2 24/2 = 12 (burst-waste excluded),
# FIXPIPE 16 -> fixpipe-bound at 16 cyc/tile.
aic_per_tile = max(16.0, 12.0)
aiv_p32 = adj["kernelB_aiv_epilogue_fp32"]["vec_total_cyc_per_element"] * 256
aiv_p16 = adj["kernelB_aiv_epilogue_fp16"]["vec_total_cyc_per_element"] * 256
ka_total = ELEMENTS * adj["kernelA"]["vec_total_cyc_per_element"]
GHZ = 1.8
us = lambda cyc: round(cyc / (GHZ * 1e3), 1)  # cycles -> us at 1.8 GHz
results["composed"] = {
    "note": "single core-pair, whole 8.39M-element problem; AIC/AIV overlap via "
            "2-slot cross-core ping-pong; ramp/drain ignored (<1000 cycles)",
    "aic_cyc_per_tile": aic_per_tile,
    "aiv_cyc_per_tile_fp32stage": round(aiv_p32, 2),
    "aiv_cyc_per_tile_fp16stage": round(aiv_p16, 2),
    "kernelA_total_cycles": round(ka_total),
    "kernelA_total_us": us(ka_total),
    "kernelB_fp32stage_total_cycles": round(max(aic_per_tile, aiv_p32) * TILES),
    "kernelB_fp32stage_total_us": us(max(aic_per_tile, aiv_p32) * TILES),
    "kernelB_fp16stage_total_cycles": round(max(aic_per_tile, aiv_p16) * TILES),
    "kernelB_fp16stage_total_us": us(max(aic_per_tile, aiv_p16) * TILES),
    "chip_level_910b": {
        "note": "910B: 24 AIC + 48 AIV. Kernel A runs on 48 AIVs; Kernel B on "
                "24 AIC+AIV pairs (24 spare AIVs remain for other work).",
        "kernelA_per_core_cycles": round(ka_total / 48),
        "kernelA_per_core_us": us(ka_total / 48),
        "kernelB_fp16_per_pair_cycles": round(max(aic_per_tile, aiv_p16) * TILES / 24),
        "kernelB_fp16_per_pair_us": us(max(aic_per_tile, aiv_p16) * TILES / 24),
    },
}
results["issue_adjustment"] = adj

with open(OUT_DIR + r"\pipesim_results.json", "w", encoding="utf-8") as f:    json.dump(results, f, indent=2)

# ---- console digest -------------------------------------------------------
print(json.dumps(results, indent=1)[:200])
for e in sweep_a:
    print(f"[A] {e['name']:24s} lim={e['limiting_stage']:14s} "
          f"steps={e['steps']:5d} cyc={e['total_cycles']:9d} "
          f"({e['total_us']:8.3f} us) stages={e['stages_cyc_per_step']}")
for e in sweep_b:
    print(f"[B] {e['name']:28s} lim={e['limiting_stage']:22s} "
          f"steps={e['steps']:5d} cyc={e['total_cycles']:9d} "
          f"({e['total_us']:8.3f} us) stages={e['stages_cyc_per_step']} "
          f"per256={e['cycles_per_256_outputs']}")
e = results["kernelB_fp16stage"]
print(f"[B] {e['name']:28s} lim={e['limiting_stage']:22s} "
      f"cyc={e['total_cycles']:9d} ({e['total_us']:8.3f} us) "
      f"stages={e['stages_cyc_per_step']} per256={round(e['total_cycles']/ELEMENTS*256,3)}")
e = results["kernelB_aiv_epilogue_fp16"]
print(f"[Bv] {e['name']:28s} lim={e['limiting_stage']:22s} cyc={e['total_cycles']:9d} "
      f"({e['total_us']:8.3f} us) stages={e['stages_cyc_per_step']}")
c = results["composed"]
print(f"\n[composed] AIC/tile={c['aic_cyc_per_tile']} cyc  "
      f"AIV/tile fp32={c['aiv_cyc_per_tile_fp32stage']}  fp16={c['aiv_cyc_per_tile_fp16stage']}")
print(f"[composed] Kernel A total = {c['kernelA_total_cycles']} cyc ({c['kernelA_total_us']} us)")
print(f"[composed] Kernel B fp32  = {c['kernelB_fp32stage_total_cycles']} cyc "
      f"({c['kernelB_fp32stage_total_us']} us)")
print(f"[composed] Kernel B fp16  = {c['kernelB_fp16stage_total_cycles']} cyc "
      f"({c['kernelB_fp16stage_total_us']} us)")
print(f"[chip] A/48cores={c['chip_level_910b']['kernelA_per_core_cycles']} cyc  "
      f"B/24pairs={c['chip_level_910b']['kernelB_fp16_per_pair_cycles']} cyc")
e = results["kernelB_aiv_epilogue"]
print(f"[Bv] {e['name']:24s} lim={e['limiting_stage']:22s} cyc={e['total_cycles']:9d} "
      f"({e['total_us']:8.3f} us) stages={e['stages_cyc_per_step']}")
print("\nviolations:", {e["name"]: e["violations"] for e in sweep_a + sweep_b
                        if e["violations"]})
print("wrote pipesim_results.json")
