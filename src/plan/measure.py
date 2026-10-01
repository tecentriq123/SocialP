import re, json, sys
sys.path.insert(0,'.')
from triage import P
s=open('/tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/gh_latest.html').read()
def strip_details(h):
    # remove nested details blocks iteratively
    while True:
        n=re.sub(r'<details\b(?:(?!<details\b).)*?</details>','',h,flags=re.S)
        if n==h: return n
        h=n
def vis(h):
    h=strip_details(h)
    h=re.sub(r'<script.*?</script>|<svg.*?</svg>|<style.*?</style>','',h,flags=re.S)
    t=re.sub(r'<[^>]+>','',h); t=re.sub(r'\s+',' ',t); return len(t)
rows=[]; unmatched=[]
for m in re.finditer(r'<template id="tpl-((?:ch|rv)\d\d)">(.*?)</template>',s,re.S):
    cid,t=m.group(1),m.group(2)
    if cid=='ch15': continue
    for sm in re.finditer(r'<section class="sec" id="([^"]+)">(.*?)</section>',t,re.S):
        sid,b=sm.group(1),sm.group(2)
        plan=P.get(sid)
        if not plan: unmatched.append(sid); continue
        parts=re.split(r'(<h3[^>]*>.*?</h3>)',b,flags=re.S)
        subs=[('(도입)',parts[0])]
        for i in range(1,len(parts),2):
            subs.append((re.sub(r'<[^>]+>','',re.sub(r'<span class="lv deep">심화</span>','',parts[i])).strip(), parts[i+1] if i+1<len(parts) else ''))
        before=sum(vis(x) for _,x in subs)
        after=0; folded=[]; moved=[]
        for name,html in subs:
            # split off practice + keypoints (always visible)
            pr=''.join(re.findall(r'<div class="practice">.*?</div>\s*</li>\s*</ol>\s*</div>|<div class="keypoints">.*?</div>',html,re.S))
            core=vis(html); prv=vis(pr) if pr else 0
            body=core-prv
            if plan['verdict']=='이동': continue
            if any(name.startswith(k) for k in plan['move']): moved.append(name); after+=prv; continue
            if name.startswith('수식으로 보기'): after+=prv; continue
            fold=plan['fold']
            if fold=='*' or fold==['*']:
                if name=='(도입)' or any(name.startswith(k) for k in plan['keep']): after+=core
                else: after+=prv; folded.append(name)
                continue
            if any(name.startswith(k) for k in fold): after+=prv; folded.append(name); continue
            if name.startswith('숫자로 따라가기') or name.startswith('숫자로 먼저'): after+=int(body*0.6)+prv; continue
            after+=core
        # check fold names matched
        names=[n for n,_ in subs]
        fl=plan['fold'] if plan['fold']!=['*'] and plan['fold']!='*' else []
        for k in list(fl)+list(plan['keep'])+list(plan['move']):
            if not any(n.startswith(k) for n in names): unmatched.append(f'{sid}:{k}')
        rows.append(dict(sid=sid,cid=cid,before=before,after=after,verdict=plan['verdict']))
json.dump(rows,open('measure.json','w'),ensure_ascii=False)
print('unmatched',unmatched)
from collections import defaultdict
tot=defaultdict(lambda:[0,0])
part=lambda c: 'P1' if c in ('ch00','ch01','ch02','ch03','ch04','ch05','ch06','ch07','rv01') else ('P2' if c<'ch14' else 'ch14')
for r in rows:
    tot[r['cid']][0]+=r['before']; tot[r['cid']][1]+=r['after']
pt=defaultdict(lambda:[0,0])
for c,(a,b) in tot.items():
    print(c,a,b,f'{b/a:.0%}'); pt[part(c)][0]+=a; pt[part(c)][1]+=b
for p,(a,b) in pt.items(): print(p,a,b,f'{b/a:.0%}')
