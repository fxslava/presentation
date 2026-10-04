"""Extract valid CA cycles from official summaries, not PV total ticks."""
import collections, gzip, json, re, tomllib
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for e in [512,1024,2048]:
    for b in [16,32]:
        for v in ['simd','cube','simt']:
            base=f'dav-3510_e2m1_b{b}_e{e}_{v}_ca'
            choices=sorted(root.glob('reports/'+base+'*'), key=lambda p:p.stat().st_mtime,reverse=True)
            chosen=None
            for report in choices:
                text=(root/'logs'/f'{report.name}.log').read_text(errors='replace') if (root/'logs'/f'{report.name}.log').exists() else ''
                status=(root/'logs'/f'{report.name}.status').read_text() if (root/'logs'/f'{report.name}.status').exists() else ''
                summary=report/'results/kernel_0_reports/summary.json'
                launches=list((report/'results').glob('kernel_*_reports'))
                if 'bit_mismatches=0 RESULT=PASS' in text and 'exit=0 ' in status and summary.exists() and len(launches)==1:
                    chosen=(report,text,status,json.loads(summary.read_text()));break
            row={'architecture':'dav-3510','soc':'Ascend950PR_9589','format':'E2M1','elements':e,'block':b,'variant':v}
            if chosen:
                report,text,status,summary=chosen
                cycles=summary['kernel_info']['kernel_total_clocks']
                row.update(cycles=cycles,cycles_per_output=cycles/e,bit_mismatches=0,
                           pipeline_utilization={k:d['mean'] for k,d in summary['pipe_utilization']['pipeline_util_summary'].items()},
                           report=report.relative_to(root).as_posix(),
                           wall_seconds=int(re.search(r'wall_seconds=(\d+)',status)[1]))
                counts=collections.Counter()
                for trace in (report/'results/kernel_0_reports').glob('core_*/trace_core*.json.gz'):
                    for event in json.load(gzip.open(trace))['traceEvents']:
                        if event.get('ph')=='X' and 'RV_VDUPS' in event.get('name',''):
                            counts['RV_VDUPS']+=1
                row['trace_broadcast_count']=counts['RV_VDUPS']
            else:row['status']='NO_VALID_COMPLETED_CA_REPORT'
            final=(root/'logs'/f'dav-3510_e2m1_b{b}_e{e}_{v}_pv_final.log')
            row['pv_final_pass']=final.exists() and 'bit_mismatches=0 RESULT=PASS' in final.read_text(errors='replace')
            rows.append(row)
pv910=[]
def flatten_blocks(tree):
    if isinstance(tree,dict):
        if 'start_cycle' in tree and 'end_cycle' in tree and 'block_id' in tree:yield tree
        for item in tree.values():yield from flatten_blocks(item)

def union_length(spans):
    end=None; total=0
    for a,b in sorted(spans):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total

for e in [512,1024,2048]:
    for b in [16,32]:
        for v in ['simd','cube','simt']:
            p=root/'logs'/f'dav-2201_e2m1_b{b}_e{e}_{v}_pv_final.log'
            if not p.exists():p=root/'logs'/f'dav-2201_e2m1_b{b}_e{e}_{v}_pv.log'
            t=p.read_text(errors='replace') if p.exists() else ''
            match=re.search(r'bit_mismatches=(\d+)',t)
            row910={'elements':e,'block':b,'variant':v,'bit_mismatches':int(match[1]) if match else None,
                          'pv_pass':bool(match and int(match[1])==0),'ca_cycles':None,
                          'status':'SIMT_BUILD_UNSUPPORTED' if v=='simt' else ('PV_PASS_NPUSIM_UNSUPPORTED' if match and int(match[1])==0 else 'PV_FAIL_NPUSIM_UNSUPPORTED')}
            cid=f'dav-2201_e2m1_b{b}_e{e}_{v}_legacy_ca'
            cl=root/'logs'/f'{cid}.log'
            profile=root/'runs'/cid/'log/profile_aiv_log0.toml'
            if cl.exists() and 'bit_mismatches=0 RESULT=PASS' in cl.read_text(errors='replace') and profile.exists():
                blocks=list(flatten_blocks(tomllib.loads(profile.read_text())))
                assert len(blocks)==1
                blk=blocks[0]
                row910.update(ca_cycles=blk['end_cycle']-blk['start_cycle'],
                              ca_core_start=blk['start_cycle'],ca_core_end=blk['end_cycle'],
                              ca_metric='raw CA profile AIV block residence cycles',
                              ca_tool='msprof op simulator',status='PV_AND_LEGACY_CA_PASS_NPUSIM_UNSUPPORTED')
                row910['ca_cycles_per_output']=row910['ca_cycles']/e
                traces=list((root/'reports'/cid).glob('OPPROF_*/simulator/core0.veccore0/trace.json'))
                if traces:
                    events=[x for x in json.loads(traces[-1].read_text())['traceEvents'] if x.get('ph')=='X' and x.get('tid') in ['SCALAR','SCALARLDST','FLOWCTRL','VECTOR','MTE2','MTE3']]
                    lo=min(x['ts'] for x in events);hi=max(x['ts']+x.get('dur',0) for x in events)
                    bypipe=collections.defaultdict(list)
                    for x in events:
                        pipe='SCALAR' if x['tid'] in ['SCALAR','SCALARLDST','FLOWCTRL'] else x['tid']
                        bypipe[pipe].append((x['ts'],x['ts']+x.get('dur',0)))
                    row910['timeline_occupancy_including_waits']={k:union_length(spans)/(hi-lo) for k,spans in bypipe.items()}
            pv910.append(row910)
result={'image':'ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12',
        'digest':'sha256:b82c08f530f956afe7c445361b1454cbcd97aaf60c4093749fe872a9657edcd7',
        'target_910b':pv910,'comparative_3510':rows,
        'note':'Pipeline percentages overlap. PV Total tick is never treated as CA core cycles.'}
(root/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'valid_ca_rows':sum('cycles' in x for x in rows),'pv950_final_passes':sum(x['pv_final_pass'] for x in rows),
                  'pv910_simd_passes':sum(x['pv_pass'] for x in pv910 if x['variant']=='simd')},indent=2))
