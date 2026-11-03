"""Offline phase-1 scoring packets, prompt contract, and response validator."""
from __future__ import annotations
import hashlib,json,os,tempfile
from collections import defaultdict
from pathlib import Path

FOCUS={"access_control","validation","arithmetic","price_calculation","state_update","external_call","transfer_amount","callback_order"}
WEIGHTS={"result_relevance":20,"asset_anomaly":15,"state_dependency":15,"parameter_influence":15,"control_interaction":15,"causal_coherence":20}
PROMPT_TEMPLATE="""# Phase 1 trace-only causal-chain scoring\nScore each supplied chain independently. CALL and ORDER are execution facts; ORDER alone is not causality. STRONG_ASSOCIATION is not proven causality and HYPOTHESIS is weak evidence only. Result confidence measures candidate-result salience, not vulnerability probability. Do not infer a vulnerability from a function name. Lower scores when evidence is incomplete and state uncertainties.\n\nReturn JSON only: {\\\"responses\\\":[{\\\"chain_id\\\":str,\\\"result_relevance\\\":0..5,\\\"asset_anomaly\\\":0..5,\\\"state_dependency\\\":0..5,\\\"parameter_influence\\\":0..5,\\\"control_interaction\\\":0..5,\\\"causal_coherence\\\":0..5,\\\"total_score\\\":0..100,\\\"concise_rationale\\\":str,\\\"key_node_ids\\\":[str],\\\"key_edge_ids\\\":[str],\\\"counter_evidence\\\":[str],\\\"uncertainties\\\":[str],\\\"source_requests\\\":[{\\\"node_id\\\":str,\\\"request_reason\\\":str,\\\"focus\\\":allowed_focus,\\\"priority\\\":1|2|3}]}]}. total_score must equal the weighted formula; do not request source beyond chain nodes.\n"""
PROMPT_TEMPLATE_V2="""# Phase 1 trace-only scoring (compact v2)\nArrays use node=[id,address_alias,function,action], edge=[edge_id,src,dst,type,evidence,...], evidence O=OBSERVED,S=STRONG_ASSOCIATION,H=HYPOTHESIS. CALL and ORDER are execution facts; ORDER is not causality. Association is not proof, H is weak evidence, result confidence is not vulnerability probability, and names alone prove nothing. Lower scores and state uncertainty when evidence is incomplete. Return the unchanged phase-1 JSON response contract with six integer scores, recomputable total_score, chain-local ids only, and at most five source_requests.\n"""

def canonical_sha256(value):
 return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def total_score(response):
 return round(sum(response[k]*w for k,w in WEIGHTS.items())/5,2)

def build_phase1_scoring_packets(graph,chains,annotations,results,transaction_graph,max_chains_per_batch=6,max_prompt_chars=60000):
 _validate_inputs(graph,chains,annotations,results,transaction_graph)
 gnodes={x['node_id']:x for x in graph['nodes']}; tx={x['node_id']:x for x in transaction_graph['nodes']}; edges={x['edge_id']:x for x in graph['edges']}
 anns={x['node_id']:x for x in annotations['node_annotations']}; rc={x['result_id']:x for x in results['result_candidates']}
 addresses=sorted({(tx[n].get('callee_address') or '').lower() for c in chains['chains'] for n in c['all_node_ids'] if tx[n].get('callee_address')})
 aliases={address:f'a{i:03d}' for i,address in enumerate(addresses,1)}
 address_dictionary={alias:address for address,alias in aliases.items()}
 node_catalog={}
 for c in chains['chains']:
  for nid in c['all_node_ids']:
   if nid not in node_catalog:
    n=tx[nid]; a=anns.get(nid,{})
    node_catalog[nid]={"node_id":nid,"address_alias":aliases.get((n.get('callee_address') or '').lower()),"function":n.get('function_name') or n.get('function_selector') or 'UNKNOWN',"action_labels":a.get('action_types',[]),"observed_operands":a.get('observed_operands',{}),"trace_index":n['trace_index'],"trace_path":n['trace_path']}
 packet_chains=[_chain_packet(c,rc[c['result_id']],edges,node_catalog) for c in chains['chains']]
 batches=[]; current=[]
 def emit(items):
  data=json.dumps({"chains":items},ensure_ascii=False,separators=(',',':'))
  body=PROMPT_TEMPLATE+'\nDATA:\n'+data
  batches.append({"batch_id":f'phase1_batch_b{len(batches)+1:03d}',"chain_ids":[x['chain_id'] for x in items],"prompt_body":body,"character_count":len(body),"estimated_tokens":(len(body)+3)//4,"token_estimate_note":"estimated as characters/4; not an exact tokenizer result"})
 for item in packet_chains:
  candidate=current+[item]
  chars=len(PROMPT_TEMPLATE+'\nDATA:\n'+json.dumps({"chains":candidate},ensure_ascii=False,separators=(',',':')))
  if current and (len(candidate)>max_chains_per_batch or chars>max_prompt_chars): emit(current);current=[item]
  elif not current and chars>max_prompt_chars: raise ValueError(f"Single chain exceeds prompt character budget: {item['chain_id']}")
  else: current=candidate
 if current:emit(current)
 return {"schema_version":"phase1_scoring_packets_v1","candidate_causal_graph_sha256":canonical_sha256(graph),"candidate_causal_chains_sha256":canonical_sha256(chains),"runtime_annotations_sha256":canonical_sha256(annotations),"result_candidates_sha256":canonical_sha256(results),"transaction_graph_sha256":canonical_sha256(transaction_graph),"address_dictionary":address_dictionary,"node_catalog":dict(sorted(node_catalog.items())),"prompt_template_sha256":hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),"batch_config":{"max_chains_per_batch":max_chains_per_batch,"max_prompt_chars":max_prompt_chars},"batches":batches}

def _chain_packet(c,r,edges,catalog):
 b=c['backbone']; main=b['node_ids']; pairs=defaultdict(list)
 for eid in b['supporting_edge_ids']:
  e=edges[eid];pairs[(e['src'],e['dst'])].append(e['edge_type'])
 def compact(eid):
  e=edges[eid]; a=e.get('attributes',{});out={"edge_id":eid,"src":e['src'],"dst":e['dst'],"edge_type":e['edge_type'],"evidence_level":e['evidence_level']}
  if a.get('relation_type'):out['relation_type']=a['relation_type']
  if e['edge_type']=='PARAM':out['matches']=(a.get('matches') or [])[:3];out['omitted_match_count']=max(0,len(a.get('matches') or [])-3)
  if e['edge_type']=='STATE':out['state_object_type']=(a.get('state_object') or {}).get('state_object_type')
  return out
 return {"chain_id":c['chain_id'],"result_id":c['result_id'],"result_type":r['result_type'],"result_confidence":r['confidence'],"semantic_views":c['semantic_views'],"backbone":{"nodes":[catalog[n] for n in main],"edges":[compact(e) for e in b['edge_ids']],"supporting_relations":[{"src":s,"dst":d,"relation_types":sorted(v),"edge_ids":sorted(e['edge_id'] for e in edges.values() if e['src']==s and e['dst']==d and e['edge_id'] in b['supporting_edge_ids'])} for (s,d),v in sorted(pairs.items())],"stop_reason":b['stop_reason']},"branches":[{"branch_id":x['branch_id'],"attachment_node_id":x['attachment_node_id'],"nodes":[catalog[n] for n in x['node_ids']],"edges":[compact(e) for e in x['edge_ids']],"evidence_level":x['evidence_level']} for x in c['branches']],"uncertainties":c['uncertainties']}

def validate_phase1_response(response,packets):
 responses=response.get('responses') if isinstance(response,dict) else response
 if not isinstance(responses,list):raise ValueError('responses must be a list')
 chain_nodes={x['chain_id']:set(x['all_node_ids']) for b in packets['batches'] for x in json.loads(b['prompt_body'].split('\nDATA:\n',1)[1])['chains'] for x in []}
 # Reconstruct allowed ids from packets' catalog and chain packet data.
 chain_data={x['chain_id']:x for b in packets['batches'] for x in json.loads(b['prompt_body'].split('\nDATA:\n',1)[1])['chains']}
 seen=set()
 for item in responses:
  required={"chain_id",*WEIGHTS,"total_score","concise_rationale","key_node_ids","key_edge_ids","counter_evidence","uncertainties","source_requests"}
  if not isinstance(item,dict) or not required<=set(item):raise ValueError('incomplete response')
  cid=item.get('chain_id');chain=chain_data.get(cid)
  if not chain or cid in seen:raise ValueError('unknown or duplicate chain_id')
  seen.add(cid)
  if any(not isinstance(item.get(k),int) or not 0<=item[k]<=5 for k in WEIGHTS):raise ValueError('scores must be integers from 0 to 5')
  if item.get('total_score')!=total_score(item):raise ValueError('incorrect total_score')
  nodes={n['node_id'] for n in chain['backbone']['nodes']}|{n['node_id'] for b in chain['branches'] for n in b['nodes']}; edges={e['edge_id'] for e in chain['backbone']['edges']}|{e for r in chain['backbone']['supporting_relations'] for e in r['edge_ids']}|{e['edge_id'] for b in chain['branches'] for e in b['edges']}
  if not set(item.get('key_node_ids',[]))<=nodes or not set(item.get('key_edge_ids',[]))<=edges:raise ValueError('response references nodes or edges outside chain')
  if not isinstance(item['concise_rationale'],str) or any(not isinstance(item[k],list) for k in ('key_node_ids','key_edge_ids','counter_evidence','uncertainties','source_requests')):raise ValueError('invalid response field type')
  req=item['source_requests']
  if len(req)>5:raise ValueError('too many source_requests')
  for q in req:
   if q.get('node_id') not in nodes or q.get('focus') not in FOCUS or q.get('priority') not in {1,2,3}:raise ValueError('invalid source_request')
  if any(k in item for k in ('ground_truth','vulnerability','vulnerability_function','line_number')):raise ValueError('forbidden response field')
 return True

def _validate_inputs(graph,chains,annotations,results,tx):
 if graph.get('schema_version')!='candidate_causal_graph_v2' or chains.get('schema_version')!='candidate_causal_chains_v1' or annotations.get('schema_version')!='runtime_annotations_v1' or results.get('schema_version')!='result_candidates_v1' or tx.get('schema_version')!='transaction_graph_v1':raise ValueError('invalid input schema')
 g={x['node_id']:(x['trace_index'],x['trace_path']) for x in graph['nodes']};t={x['node_id']:(x['trace_index'],x['trace_path']) for x in tx['nodes']}
 if g!=t:raise ValueError('transaction graph node_id/trace_index/trace_path mismatch')

def serialize_packets(p):return json.dumps(p,ensure_ascii=False,indent=2)+'\n'
def write_text(text,path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);h=tempfile.NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False)
 try:
  with h:h.write(text)
  os.replace(h.name,p)
 except Exception:Path(h.name).unlink(missing_ok=True);raise

def heuristic_tokens(text):
 ascii_count=sum(ord(c)<128 for c in text);return (ascii_count+2)//3+(len(text)-ascii_count)

def build_phase1_scoring_packets_v2(graph,chains,annotations,results,transaction_graph,max_chains_per_batch=6,max_prompt_chars=60000):
 _validate_inputs(graph,chains,annotations,results,transaction_graph)
 tx={n['node_id']:n for n in transaction_graph['nodes']};ann={n['node_id']:n for n in annotations['node_annotations']};edge={e['edge_id']:e for e in graph['edges']};rc={r['result_id']:r for r in results['result_candidates']}
 addresses=sorted({(tx[n].get('callee_address') or '').lower() for c in chains['chains'] for n in c['all_node_ids'] if tx[n].get('callee_address')});aliases={a:f'a{i:03d}' for i,a in enumerate(addresses,1)}
 resolver={nid:{'address':tx[nid].get('callee_address'),'function':tx[nid].get('function_name') or tx[nid].get('function_selector') or 'UNKNOWN','trace_index':tx[nid]['trace_index'],'trace_path':tx[nid]['trace_path']} for c in chains['chains'] for nid in c['all_node_ids']}
 resolver=dict(sorted(resolver.items()))
 def enc_edge(eid):
  e=edge[eid];a=e.get('attributes',{});tail=[]
  if e['edge_type']=='PARAM':
   m=[ [x.get('value_type'),x.get('normalized_value')] for x in (a.get('matches') or [])[:2]];tail=[a.get('relation_type'),m,max(0,len(a.get('matches') or [])-2)]
  elif e['edge_type']=='STATE':tail=[a.get('relation_type'),(a.get('state_object') or {}).get('state_object_type')]
  elif e['edge_type']=='ASSET':tail=[a.get('asset_address',{}).get('normalized'),a.get('match_type'),a.get('allocated_amount_raw')]
  elif a.get('relation_type'):tail=[a['relation_type']]
  return [eid,e['src'],e['dst'],e['edge_type'],{'OBSERVED':'O','STRONG_ASSOCIATION':'S','HYPOTHESIS':'H'}[e['evidence_level']],*tail]
 def enc_chain(c):
  b=c['backbone'];pairs=defaultdict(list)
  for eid in b['supporting_edge_ids']:
   e=edge[eid];pairs[(e['src'],e['dst'])].append([eid,e['edge_type'],{'OBSERVED':'O','STRONG_ASSOCIATION':'S','HYPOTHESIS':'H'}[e['evidence_level']]])
  return [c['chain_id'],c['result_id'],rc[c['result_id']]['result_type'],rc[c['result_id']]['confidence'],b['node_ids'],b['edge_ids'],[[s,d,v] for (s,d),v in sorted(pairs.items())],[[x['branch_id'],x['attachment_node_id'],x['node_ids'],x['edge_ids'],x['semantic_edge_id'],{'OBSERVED':'O','STRONG_ASSOCIATION':'S','HYPOTHESIS':'H'}[x['evidence_level']]] for x in c['branches']],list(dict.fromkeys(c['uncertainties']))]
 groups=defaultdict(list)
 for c in chains['chains']:groups[c['result_id']].append(c)
 batches=[];current=[]
 def body(items):
  ids={n for c in items for n in c['all_node_ids']};eids={e for c in items for e in c['all_edge_ids']};payload={'nodes':[[n,aliases.get((tx[n].get('callee_address') or '').lower()),tx[n].get('function_name') or tx[n].get('function_selector') or 'UNKNOWN',(ann.get(n,{}).get('action_types') or [''])[0],ann.get(n,{}).get('observed_operands',{})] for n in sorted(ids)],'edges':[enc_edge(e) for e in sorted(eids)],'chains':[enc_chain(c) for c in items]};text=PROMPT_TEMPLATE_V2+'\nDATA:'+json.dumps(payload,ensure_ascii=False,separators=(',',':'));return payload,text
 for rid in [r['result_id'] for r in results['result_candidates']]:
  if rid not in groups:continue
  cand=current+groups[rid];_,text=body(cand)
  if current and (len(cand)>max_chains_per_batch or len(text)>max_prompt_chars):
   payload,text=body(current);batches.append({'batch_id':f'phase1_batch_b{len(batches)+1:03d}','chain_ids':[c['chain_id'] for c in current],'llm_payload':payload,'character_count':len(text),'estimated_tokens':heuristic_tokens(text),'token_estimate_note':'heuristic ceil(ASCII/3)+non-ASCII; not exact'});current=groups[rid]
  elif not current and len(text)>max_prompt_chars:raise ValueError(f'Single result group exceeds prompt budget: {rid}')
  else:current=cand
 if current:
  payload,text=body(current);batches.append({'batch_id':f'phase1_batch_b{len(batches)+1:03d}','chain_ids':[c['chain_id'] for c in current],'llm_payload':payload,'character_count':len(text),'estimated_tokens':heuristic_tokens(text),'token_estimate_note':'heuristic ceil(ASCII/3)+non-ASCII; not exact'})
 allids=[x for b in batches for x in b['chain_ids']]
 if len(allids)!=len(chains['chains']) or len(allids)!=len(set(allids)):raise ValueError('chain coverage is not exactly once')
 return {'schema_version':'phase1_scoring_packets_v2','input_sha256':{'transaction_graph_v1':canonical_sha256(transaction_graph),'runtime_annotations_v1':canonical_sha256(annotations),'result_candidates_v1':canonical_sha256(results),'candidate_causal_graph_v2':canonical_sha256(graph),'candidate_causal_chains_v1':canonical_sha256(chains)},'resolver_metadata':{'address_dictionary':{v:k for k,v in aliases.items()},'nodes':resolver},'prompt_template_sha256':hashlib.sha256(PROMPT_TEMPLATE_V2.encode()).hexdigest(),'batch_config':{'max_chains_per_batch':max_chains_per_batch,'max_prompt_chars':max_prompt_chars},'batches':batches}
