import gzip, json, pathlib, collections
root = pathlib.Path('/work/dual_kernel_study/cann_audit/native950')
result = {}
for label, directory in [('A','report_A'), ('B','report_B_nomap'), ('stage32','report_stage32')]:
    p = root / directory / 'results/kernel_0_reports'
    if not p.exists(): continue
    summary = json.loads((p/'summary.json').read_text())
    events = json.load(gzip.open(p/'core_0/trace_core0.json.gz'))['traceEvents']
    starts = {}; slacks = collections.defaultdict(list)
    for e in sorted(events, key=lambda x:x.get('ts',0)):
        if e.get('cat') != 'Pipe dependency (flags)': continue
        if e['ph'] == 's': starts[e['id']] = e
        elif e['ph'] == 'f' and e['id'] in starts:
            first = starts.pop(e['id'])
            slacks[first['name']].append(round((e['ts']-first['ts'])*1650))
    summary['elements'] = 8192
    summary['cycles_per_element'] = summary['kernel_info']['kernel_total_clocks']/8192
    summary['flag_flow_elapsed_cycles'] = {k: {'count':len(v),'sum':sum(v),'min':min(v),'max':max(v)} for k,v in slacks.items()}
    summary['flag_flow_note'] = 'Elapsed SetFlag-to-WaitFlag flow spans, not exclusive stall cycles; overlapping spans must not be summed as kernel overhead. Trace microseconds converted at simulator 1650 MHz.'
    result[label] = summary
(root/'simulation_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{'cycles':v['kernel_info']['kernel_total_clocks'],'cycles_per_element':v['cycles_per_element'],'flag_flows':v['flag_flow_elapsed_cycles']} for k,v in result.items()},indent=2))
