"""Offline validation, ranking, and deterministic diversity selection for phase 1."""
from __future__ import annotations
import hashlib,json,os,tempfile
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from .phase1_scoring import FOCUS,WEIGHTS,total_score,canonical_sha256

Q=Decimal('0.0001')
def _d(v):return Decimal(str(v)).quantize(Q,rounding=ROUND_HALF_UP)
def _jac(a,b):return Decimal(len(a&b))/Decimal(len(a|b)) if a|b else Decimal(0)
def _decode_packets(packets):
 chains={};edges={};batch_ids=set()
 for batch in packets['batches']:
  if batch['batch_id'] in batch_ids:raise ValueError('duplicate batch_id')
  batch_ids.add(batch['batch_id']);payload=batch['llm_payload']
  edges.update({e[0]:e for e in payload['edges']})
  for c in payload['chains']:
   # [chain,result,type,confidence,main_nodes,main_edges,support,branches,uncertainties]
   nodes=set(c[4]);eids=set(c[5]);
   for s in c[6]:eids.update(x[0] for x in s[2])
   for br in c[7]:nodes.update(br[2]);eids.update(br[3])
   chains[c[0]]={'batch_id':batch['batch_id'],'result_id':c[1],'nodes':nodes,'edges':eids,'uncertainties':c[8]}
 return chains,edges,batch_ids
def validate_and_rank(packets,chains_document,batch_responses,k=5,lam='0.90'):
 if packets.get('schema_version')!='phase1_scoring_packets_v2' or chains_document.get('schema_version')!='candidate_causal_chains_v1':raise ValueError('invalid input schema')
 chains,edges,batch_ids=_decode_packets(packets);expected={c['chain_id'] for c in chains_document['chains']}
 if set(chains)!=expected:raise ValueError('packet chain coverage mismatch')
 seen_batches=set();responses=[]
 for doc in batch_responses:
  bid=doc.get('batch_id');bid=f'phase1_batch_{bid}' if isinstance(bid,str) and bid.startswith('b') else bid
  if bid not in batch_ids or bid in seen_batches:raise ValueError('unknown or duplicate batch_id')
  seen_batches.add(bid)
  for r in doc.get('responses',[]):
   r=dict(r);r.update(r.pop('scores',{}))
   cid=r.get('chain_id');meta=chains.get(cid)
   if not meta or meta['batch_id']!=bid:raise ValueError('unknown or cross-batch chain_id')
   required={'chain_id',*WEIGHTS,'total_score','concise_rationale','key_node_ids','key_edge_ids','counter_evidence','uncertainties','source_requests'}
   if not required<=set(r) or any(not isinstance(r[x],int) or not 0<=r[x]<=5 for x in WEIGHTS):raise ValueError('invalid response scores')
   if Decimal(str(r['total_score']))!=Decimal(str(total_score(r))):raise ValueError('incorrect total_score')
   if not set(r['key_node_ids'])<=meta['nodes'] or not set(r['key_edge_ids'])<=meta['edges']:raise ValueError('response references outside chain')
   if len(r['source_requests'])>5:raise ValueError('too many source_requests')
   for q in r['source_requests']:
    if q.get('node_id') not in meta['nodes'] or q.get('focus') not in FOCUS or q.get('priority') not in {1,2,3}:raise ValueError('invalid source_request')
   responses.append(dict(r))
 if seen_batches!=batch_ids:raise ValueError('missing batch response')
 if {r['chain_id'] for r in responses}!=expected or len(responses)!=len(expected):raise ValueError('missing or duplicate chain_id')
 raw=sorted(responses,key=lambda r:(-Decimal(str(r['total_score'])),-r['causal_coherence'],-r['result_relevance'],r['chain_id']))
 for i,r in enumerate(raw,1):r['raw_rank']=i
 sig={cid:{(edges[e][3],edges[e][5] if len(edges[e])>5 and isinstance(edges[e][5],str) else None) for e in m['edges']} for cid,m in chains.items()}
 lam=Decimal(str(lam));selected=[];remaining=list(raw)
 while remaining and len(selected)<min(k,len(raw)):
  if not selected:pick=remaining[0];sim=Decimal(0);near=None;reason='RAW_TOP_1_FIXED'
  else:
   scored=[]
   for r in remaining:
    sims=[Decimal('0.6')*_jac(chains[r['chain_id']]['nodes'],chains[s['chain_id']]['nodes'])+Decimal('0.3')*_jac(sig[r['chain_id']],sig[s['chain_id']])+Decimal('0.1')*(chains[r['chain_id']]['result_id']==chains[s['chain_id']]['result_id']) for s in selected]
    mx=max(sims);near=selected[sims.index(mx)]['chain_id'];mmr=lam*Decimal(str(r['total_score']))/100-(Decimal(1)-lam)*mx;scored.append((mmr,r,mx,near))
   scored.sort(key=lambda x:(-x[0],-Decimal(str(x[1]['total_score'])),-x[1]['causal_coherence'],x[1]['raw_rank'],x[1]['chain_id']));_,pick,sim,near=scored[0];reason='MMR_DIVERSITY_SELECTION'
  pick=dict(pick);pick.update({'mmr_score':str(_d(Decimal(str(pick['total_score']))/100 if not selected else (lam*Decimal(str(pick['total_score']))/100-(Decimal(1)-lam)*sim))),'max_similarity':str(_d(sim)),'most_similar_chain_id':near,'selection_reason':reason});selected.append(pick);remaining=[r for r in remaining if r['chain_id']!=pick['chain_id']]
 return {'schema_version':'phase1_ranked_chains_v1','input_sha256':{'phase1_scoring_packets_v2':canonical_sha256(packets),'candidate_causal_chains_v1':canonical_sha256(chains_document),'batch_responses':canonical_sha256(batch_responses)},'scoring_config':{'k':k,'lambda':str(lam),'similarity':'0.6*node_jaccard + 0.3*semantic_signature_jaccard + 0.1*same_result'},'raw_ranking':raw,'selected_topk':selected,'selection_audit':{'chain_count':len(expected),'response_count':len(responses),'all_chains_scored_exactly_once':True}}
def write_ranked(result,path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False)
 try:
  with h:h.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  os.replace(h.name,p)
 except Exception:Path(h.name).unlink(missing_ok=True);raise
