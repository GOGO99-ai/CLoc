from __future__ import annotations
from dataclasses import dataclass,field,asdict
from typing import Any
@dataclass(frozen=True)
class SourceRequest:
 chain:str;address:str;tx_hash:str|None=None;origins:tuple[dict[str,Any],...]=()
@dataclass
class SourceResult:
 chain:str;requested_address:str;resolved_address:str;status:str;verified:bool=False;cache_hit:bool=False;source_format:str='unknown';contract_name:str='Unknown';files:list[dict[str,Any]]=field(default_factory=list);abi_path:str|None=None;proxy:dict[str,Any]=field(default_factory=dict);failure:dict[str,Any]|None=None;attempt_count:int=0;origins:list[dict[str,Any]]=field(default_factory=list);artifact_hashes:dict[str,str]=field(default_factory=dict)
 def to_dict(self):return asdict(self)
