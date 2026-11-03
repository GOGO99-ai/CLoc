from __future__ import annotations
import re
from .graph_models import TransactionGraph
from .parameter_evidence_models import ParamAtom
from .runtime_semantics import split_top_level_arguments
ADDR=re.compile(r"^0x[0-9a-fA-F]{40}$"); UINT=re.compile(r"^(\d+)(?:\s+\[[^\]]+\])?$")
def extract_parameter_atoms(graph:TransactionGraph)->list[ParamAtom]:
    out=[]
    for node in graph.nodes:
        for i,raw in enumerate(split_top_level_arguments(node.params_raw)):_walk(node,raw,f"arg[{i}]",out)
    return sorted(out,key=lambda x:(x.trace_index,x.argument_path,x.atom_type,x.normalized_value))
def _walk(node,raw,path,out):
    value=raw.strip()
    if len(value)>=2 and value[0]=='[' and value[-1]==']':
        inner=value[1:-1].strip()
        if inner:
            for i,item in enumerate(split_top_level_arguments(inner)):_walk(node,item,f"{path}[{i}]",out)
        return
    if ADDR.fullmatch(value): typ,norm,rule="ADDRESS",value.lower(),"FULL_20_BYTE_ADDRESS"
    else:
        m=UINT.fullmatch(value)
        if not m:return
        typ,norm,rule="UINT",m.group(1),"TOP_LEVEL_DECIMAL_UINT"
    out.append(ParamAtom(node.node_id,path,typ,value,norm,rule,(node.trace_path,),node.trace_index))
