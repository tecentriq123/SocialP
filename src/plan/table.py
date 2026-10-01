import json,re,sys
from triage import P
rows={r['sid']:r for r in json.load(open('measure.json'))}
st=json.load(open('/tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/structure_titles.json')) if False else None
s=open('/tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/gh_latest.html').read()
titles={}
for m in re.finditer(r'<section class="sec" id="([^"]+)">.*?<h2[^>]*>(.*?)</h2>',s,re.S):
    t=re.sub(r'<[^>]+>','',m.group(2)).strip(); titles[m.group(1)]=t
VI={'필수':0,'축소':1,'선택':2,'이동':3}
def label(sid):
    c,sn=sid.split('-'); t=titles[sid]
    m=re.match(r'(.)\.(.*)',t); let,name=(m.group(1),m.group(2).strip()) if m else ('',t)
    ch='분석 고르기 연습' if c=='rv01' else f'{int(c[2:])}장'
    return f'{ch} {let}. {name}'
def table(ids,key0):
    out=['| 절 | 판정 | 접거나 옮길 부분 | 메모 | 분량(천 자, 현재 → 개편) |','| --- | --- | --- | --- | --- |']; blocks={}
    for i,sid in enumerate(ids):
        p=P[sid]; r=rows.get(sid)
        if p['fold'] in ('*',['*']):
            f='전부' + (f" (남김: {', '.join(p['keep'])})" if p['keep'] else '')
        else: f=', '.join(p['fold'])
        if p['move']: f=(f+'; ' if f else '')+'; '.join(f'{k} → {v}' for k,v in p['move'].items())
        if not f: f='–'
        k=f'{key0}{i}'; blocks[k]={'type':'dropdown','enum':'$lid:v' if key0=='a' else 'ENUM','index':VI[p['verdict']]}
        amt=f"{r['before']/1000:.1f} → {r['after']/1000:.1f}" if r else '–'
        note=p['note'] or '–'
        out.append(f"| {label(sid)} | <?claude block {k}?> | {f.replace('|','/')} | {note} | {amt} |")
    return '\n'.join(out),blocks
ids=[sid for sid in P if sid.split('-')[0] in ('ch00','ch01','ch02','ch03','ch04','ch05','ch06','ch07','rv01')]
# order: rv01 after ch07 already by insertion order
md,blocks=table(ids,'a')
json.dump({'md':md,'blocks':blocks},open('p1.json','w'),ensure_ascii=False)
print(md); print(len(blocks))
