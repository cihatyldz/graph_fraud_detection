import json,sys,re
nb=json.load(open(sys.argv[1]))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    src=''.join(c['source']).splitlines()
    print(f"=== [{i}] ec={c.get('execution_count')} | {src[0][:70] if src else ''}")
    for o in c.get('outputs',[]):
        t=o.get('text') or o.get('data',{}).get('text/plain') or ''
        if o.get('output_type')=='error': t='ERROR '+o['ename']+': '+o['evalue']
        t=''.join(t) if isinstance(t,list) else t
        print(re.sub(r'\x1b\[[0-9;]*m','',t))
