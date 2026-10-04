import gzip
import json
from pathlib import Path

ROOT = Path('/work/dual_kernel_study/cann_audit/native950')
REPORTS = {
    ('A', 16): 'report_simd_b16',
    ('A', 32): 'report_simd_b32',
    ('A', 64): 'report_simd_b64',
    ('B', 16): 'report_simt_direct_ub_b16',
    ('B', 32): 'report_simt_direct_ub_b32',
    ('B', 64): 'report_simt_direct_ub_b64',
    ('C', 16): 'report_simt_hybrid_clean_b16',
    ('C', 32): 'report_simt_hybrid_clean_b32',
    ('C', 64): 'report_simt_hybrid_clean_b64',
}
def explicit_ub_bytes(path, block):
    if path == 'A':
        return 66432 + 2*4096 + 2*(8192//block) + 2*16384
    return 16384 if path == 'B' else 0
study = {'target': 'Ascend950PR_9589 / dav-3510 / CANN 9.2.0-beta.2',
         'elements': 8192, 'block_dim': 1,
         'caveats': ['UB allocation is explicit source allocation, not measured peak occupancy.',
                     'Broadcast slot cycles overlap and are not exclusive execution time.',
                     'No UB bank conflict or per-instruction GM latency counter exported.'],
         'paths': {}}
for (path, block), name in REPORTS.items():
    report = ROOT/name/'results/kernel_0_reports'
    if not (report/'summary.json').exists():
        continue
    summary = json.loads((report/'summary.json').read_text())
    trace_path = report/'core_0/trace_core0.json'
    if trace_path.exists():
        events = json.loads(trace_path.read_text())['traceEvents']
    else:
        with gzip.open(str(trace_path)+'.gz') as f:
            events = json.load(f)['traceEvents']
    dups = [e for e in events if e.get('name') == 'RV_VDUPS']
    vf_report_path = report/'kernel_launch_0_vf_report.json'
    vfs = json.loads(vf_report_path.read_text())['vfs'] if vf_report_path.exists() else []
    cycles = summary['kernel_info']['kernel_total_clocks']
    entry = {'cycles': cycles, 'cycles_per_output': cycles/8192,
             'broadcast_instructions': len(dups),
             'broadcast_pipe_slot_cycles': round(sum(e.get('dur',0)*1650 for e in dups)),
             'explicit_ub_bytes': explicit_ub_bytes(path, block),
             'explicit_gm_workspace_bytes': 49152 if path == 'C' else 0,
             'pipeline_utilization': {k: v['mean'] for k,v in summary['pipe_utilization']['pipeline_util_summary'].items()},
             'simt_vfs': [{'executions': vf['executions'],
                           'avg_cycles': vf['timing']['avg_cycles'],
                           'pc': vf['function_pc']}
                          for vf in vfs],
             'aiv_vector_instructions': summary.get('aiv_vector_instructions', {}),
             'report': str(report.parent.parent)}
    study['paths'][f'{path}_block{block}'] = entry
for block in (16,32,64):
    a = study['paths'].get(f'A_block{block}')
    b = study['paths'].get(f'B_block{block}')
    c = study['paths'].get(f'C_block{block}')
    if a and b: b['speedup_vs_A'] = round(a['cycles']/b['cycles'],4)
    if a and c: c['speedup_vs_A'] = round(a['cycles']/c['cycles'],4)
study['matrix_4096x4096_linear_projection'] = {
    name: value['cycles']*2048 for name,value in study['paths'].items()}
study['matrix_4096x4096_projection_note'] = 'Arithmetic extrapolation from a single 8192-output tile; no full-matrix CA run or multicore model.'
for probe in ('scalar','simt'):
    p = ROOT/f'report_{probe}_switch/results/kernel_0_reports/summary.json'
    if p.exists():
        study.setdefault('switch_probe',{})[probe] = json.loads(p.read_text())['kernel_info']['kernel_total_clocks']
if len(study.get('switch_probe',{})) == 2:
    study['switch_probe']['observed_incremental_cycles'] = study['switch_probe']['simt']-study['switch_probe']['scalar']
    study['switch_probe']['note'] = 'Minimal-store kernel difference, not isolated hardware VF switch latency.'
direct = ROOT/'report_simt_direct/results/kernel_0_reports/summary.json'
if direct.exists():
    study['gm_only_variant_block16_cycles'] = json.loads(direct.read_text())['kernel_info']['kernel_total_clocks']
(ROOT/'simt_study_results.json').write_text(json.dumps(study,indent=2))
for name, v in study['paths'].items():
    print(name, v['cycles'], v['broadcast_instructions'], v['broadcast_pipe_slot_cycles'], v.get('speedup_vs_A'))
