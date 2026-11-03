"""Post-freeze localization ground-truth resolution and three-level Top-K evaluation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .source_index import parse_solidity_source
from .verified_source_cache_index import VerifiedSourceCacheIndex, lookup_verified_source_cache

_ADDRESS=re.compile(r"^0x[0-9a-fA-F]{40}$")
_LOCATION=re.compile(r"^(0x[0-9a-fA-F]{40})#([^#]*)#([0-9]+):([0-9]+)$")


@dataclass(frozen=True)
class GroundTruthResolutionIssue:
    code: str


@dataclass(frozen=True)
class LocalizationGroundTruthLabel:
    blockchain_chain_id: int
    source_file: str | None
    offset: int
    length: int
    reason_codes: tuple[str,...]=()
    contract_address: str=field(default="",repr=False)
    raw_location: str=field(default="",repr=False)
    source_file_suffixes: tuple[str,...]=field(default=(),repr=False)
    absolute_path_recovered: bool=field(default=False,repr=False)

    def public_summary(self)->dict[str,Any]:
        return {"blockchain_chain_id":self.blockchain_chain_id,"source_file_reported":self.source_file is not None,"offset":self.offset,"length":self.length,"reason_codes":list(self.reason_codes)}


@dataclass(frozen=True)
class ResolvedLocalizationGroundTruth:
    status: str
    source_file_available: bool
    function_available: bool
    line_range_available: bool
    truth_start_line: int|None
    truth_end_line: int|None
    function_signature: str|None
    function_name: str|None
    offset_mode: str="UTF8_BYTE_OFFSET"
    reason_codes: tuple[str,...]=()
    normalized_source_file: str|None=field(default=None,repr=False)
    function_name_unique_in_file: bool=field(default=False,repr=False)
    source_binding_role: str|None=field(default=None,repr=False)
    path_recovery_applied: bool=field(default=False,repr=False)

    def public_summary(self)->dict[str,Any]:
        return {"status":self.status,"source_file_available":self.source_file_available,"function_available":self.function_available,"line_range_available":self.line_range_available,"offset_mode":self.offset_mode,"reason_codes":list(self.reason_codes)}


@dataclass(frozen=True)
class LocalizationTopKMetrics:
    file_recall_at_1: bool|None
    file_recall_at_3: bool|None
    function_recall_at_1: bool|None
    function_recall_at_3: bool|None
    line_range_recall_at_1: bool|None
    line_range_recall_at_3: bool|None
    topk_generated: bool

    def to_dict(self)->dict[str,Any]:
        return dict(self.__dict__)


def _normal_path(value:str)->str:
    value=value.replace("\\","/").strip()
    path=PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:",value): raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    return path.as_posix().lstrip("./")


def parse_localization_ground_truth(location:Any,*,blockchain_chain_id:int=1)->LocalizationGroundTruthLabel:
    if not isinstance(location,str): raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    match=_LOCATION.fullmatch(location.strip())
    if not match: raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    address,file_value,offset_text,length_text=match.groups(); offset=int(offset_text); length=int(length_text)
    if not _ADDRESS.fullmatch(address) or offset<0 or length<=0: raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    reasons=[]; source_file=None
    if file_value:
        source_file=_normal_path(file_value)
    else: reasons.append("GROUND_TRUTH_FILE_NOT_REPORTED")
    suffixes=(source_file,) if source_file else ()
    return LocalizationGroundTruthLabel(blockchain_chain_id,source_file,offset,length,tuple(reasons),address.lower(),location,suffixes,False)


_ABSOLUTE_DRIVE=re.compile(r"^[A-Za-z]:/")
_PATH_MARKERS=frozenset({"contracts","src","sources"})
GROUND_TRUTH_SOURCE_BINDING_ROLES=(
    "GROUND_TRUTH_LOCATION_ADDRESS",
    "RUNTIME_ADDRESS",
    "EFFECTIVE_IMPLEMENTATION_ADDRESS",
    "PROXY_IMPLEMENTATION_ADDRESS",
    "REQUESTED_SOURCE_ADDRESS",
)


def parse_localization_ground_truth_repair_v1(location:Any,*,blockchain_chain_id:int=1)->LocalizationGroundTruthLabel:
    """Parse an evaluation label with a non-dereferenced absolute-path suffix.

    The original absolute path is never opened.  Every allowed marker suffix is
    retained internally so the resolver can require a unique cache-file match.
    """
    if not isinstance(location,str): raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    match=_LOCATION.fullmatch(location.strip())
    if not match: raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    address,file_value,offset_text,length_text=match.groups();offset=int(offset_text);length=int(length_text)
    if not _ADDRESS.fullmatch(address) or offset<0 or length<=0: raise ValueError("GROUND_TRUTH_LOCATION_INVALID")
    if not file_value:
        return LocalizationGroundTruthLabel(blockchain_chain_id,None,offset,length,("GROUND_TRUTH_FILE_NOT_REPORTED",),address.lower(),location,(),False)
    normalized=file_value.replace("\\","/").strip()
    is_absolute=normalized.startswith("/") or normalized.startswith("//") or _ABSOLUTE_DRIVE.match(normalized) is not None
    if not is_absolute:
        return parse_localization_ground_truth(location,blockchain_chain_id=blockchain_chain_id)
    parts=tuple(part for part in normalized.split("/") if part not in {"","."})
    if ".." in parts: raise ValueError("GROUND_TRUTH_PATH_TRAVERSAL_REJECTED")
    positions=tuple(index for index,part in enumerate(parts) if part.lower() in _PATH_MARKERS)
    suffixes=tuple(dict.fromkeys(PurePosixPath(*parts[index:]).as_posix() for index in positions if index+1<len(parts)))
    if not suffixes: raise ValueError("GROUND_TRUTH_PATH_SUFFIX_NOT_ALLOWED")
    return LocalizationGroundTruthLabel(blockchain_chain_id,suffixes[0],offset,length,("GROUND_TRUTH_ABSOLUTE_PATH_SUFFIX_RECOVERED",),address.lower(),location,suffixes,True)


def _utf8_lines(text:str,offset:int,length:int)->tuple[int,int]:
    raw=text.encode("utf-8"); end=offset+length
    if offset<0 or end>len(raw): raise ValueError("GROUND_TRUTH_OFFSET_OUT_OF_RANGE")
    try: raw[:offset].decode("utf-8"); raw[offset:end].decode("utf-8"); raw[end:].decode("utf-8")
    except UnicodeDecodeError: raise ValueError("GROUND_TRUTH_OFFSET_ENCODING_INVALID") from None
    start_line=raw[:offset].count(b"\n")+1
    end_position=max(offset,end-1); end_line=raw[:end_position].count(b"\n")+1
    return start_line,end_line


def resolve_localization_ground_truth(label:LocalizationGroundTruthLabel,index:VerifiedSourceCacheIndex)->ResolvedLocalizationGroundTruth:
    lookup=lookup_verified_source_cache(index,runtime_address=label.contract_address)
    if lookup.status=="VERIFIED_SOURCE_CACHE_CONFLICT": return _failed("GROUND_TRUTH_SOURCE_CACHE_CONFLICT")
    if lookup.status!="VERIFIED_SOURCE_CACHE_HIT" or lookup.binding is None: return _failed("GROUND_TRUTH_SOURCE_CACHE_MISSING")
    artifact=index.cache_root/lookup.binding.artifact_relative_path; manifest_path=artifact/"manifest.json"
    try: manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError): return _failed("GROUND_TRUTH_SOURCE_CACHE_MISSING")
    candidates=[]
    for item in manifest.get("files",[]):
        if not isinstance(item,Mapping) or not isinstance(item.get("path"),str): continue
        try: normalized=_normal_path(item["path"]); path=(artifact/normalized).resolve(); path.relative_to(artifact.resolve()); text=path.read_text(encoding="utf-8")
        except (ValueError,OSError,UnicodeError): continue
        candidates.append((normalized,path,text))
    if label.source_file:
        exact=[x for x in candidates if x[0]==label.source_file or x[0].endswith("/"+label.source_file)]
        if not exact:
            name=PurePosixPath(label.source_file).name; exact=[x for x in candidates if PurePosixPath(x[0]).name==name]
        if not exact: return _failed("GROUND_TRUTH_FILE_NOT_FOUND")
        if len(exact)!=1: return _failed("GROUND_TRUTH_FILE_AMBIGUOUS")
        chosen=exact[0]
    else:
        fitting=[]
        for item in candidates:
            try:_utf8_lines(item[2],label.offset,label.length); fitting.append(item)
            except ValueError: pass
        if not fitting: return _failed("GROUND_TRUTH_OFFSET_OUT_OF_RANGE")
        if len(fitting)!=1: return _failed("GROUND_TRUTH_FILE_AMBIGUOUS")
        chosen=fitting[0]
    try:start_line,end_line=_utf8_lines(chosen[2],label.offset,label.length)
    except ValueError as exc:return _failed(str(exc))
    parsed=parse_solidity_source(chosen[2]); functions=[f for f in parsed.get("functions",[]) if f.get("signature_start_line")<=start_line and f.get("end_line")>=end_line]
    reasons=[]; signature=name=None
    if len(functions)==1: signature=functions[0].get("canonical_signature") or functions[0].get("function_name"); name=functions[0].get("function_name")
    elif len(functions)>1: reasons.append("GROUND_TRUTH_FUNCTION_AMBIGUOUS")
    else: reasons.append("GROUND_TRUTH_FUNCTION_NOT_FOUND")
    status="RESOLVED_FUNCTION" if signature else "RESOLVED_FILE_AND_LINES"
    same_name=sum(1 for item in parsed.get("functions",[]) if item.get("function_name")==name) if name else 0
    return ResolvedLocalizationGroundTruth(status,True,bool(signature),True,start_line,end_line,signature,name,"UTF8_BYTE_OFFSET",tuple(reasons),chosen[0],same_name==1)


def _evaluation_role_bindings(index:VerifiedSourceCacheIndex,address:str,role:str)->tuple[Any,...]:
    if role not in GROUND_TRUTH_SOURCE_BINDING_ROLES: raise ValueError("GROUND_TRUTH_SOURCE_BINDING_ROLE_INVALID")
    rows=[]
    for item in index.bindings:
        direct=item.proxy_resolution_source=="DIRECT_VERIFIED_CACHE"
        proxy=item.proxy_resolution_source=="VERIFIED_PROXY_IMPLEMENTATION"
        matched=(
            role=="GROUND_TRUTH_LOCATION_ADDRESS" and direct and item.artifact_address==address
            or role=="RUNTIME_ADDRESS" and direct and item.runtime_address==address
            or role=="EFFECTIVE_IMPLEMENTATION_ADDRESS" and proxy and item.effective_address==address
            or role=="PROXY_IMPLEMENTATION_ADDRESS" and proxy and item.runtime_address==address
            or role=="REQUESTED_SOURCE_ADDRESS" and direct and item.requested_address==address
        )
        if matched: rows.append(item)
    unique={(item.source_sha256,item.artifact_relative_path):item for item in rows}
    return tuple(sorted(unique.values(),key=lambda item:(item.source_sha256,item.artifact_relative_path)))


def _binding_source_files(index:VerifiedSourceCacheIndex,binding:Any)->tuple[tuple[str,str],...]:
    artifact=(index.cache_root/binding.artifact_relative_path).resolve()
    try: artifact.relative_to(index.cache_root.resolve())
    except ValueError:return ()
    try:manifest=json.loads((artifact/"manifest.json").read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):return ()
    rows=[]
    for item in manifest.get("files",[]):
        if not isinstance(item,Mapping) or not isinstance(item.get("path"),str):continue
        try:
            normalized=_normal_path(item["path"]);path=(artifact/normalized).resolve();path.relative_to(artifact);text=path.read_text(encoding="utf-8")
        except (ValueError,OSError,UnicodeError):continue
        rows.append((normalized,text))
    return tuple(rows)


def _repair_failed(label:LocalizationGroundTruthLabel,code:str,role:str|None=None)->ResolvedLocalizationGroundTruth:
    reasons=tuple(dict.fromkeys((*label.reason_codes,code)))
    return ResolvedLocalizationGroundTruth("UNRESOLVED",False,False,False,None,None,None,None,"UTF8_BYTE_OFFSET",reasons,None,False,role,label.absolute_path_recovered)


def resolve_localization_ground_truth_repair_v1(label:LocalizationGroundTruthLabel,index:VerifiedSourceCacheIndex)->ResolvedLocalizationGroundTruth:
    """Resolve truth through role-separated, verified cache bindings only.

    Roles are visited in a fixed order.  Bindings from different roles are
    never mixed, and a role containing multiple source digests is rejected.
    Prediction files and candidate rankings are not accepted by this API.
    """
    if not isinstance(label,LocalizationGroundTruthLabel) or not isinstance(index,VerifiedSourceCacheIndex):
        raise ValueError("GROUND_TRUTH_REPAIR_INPUT_INVALID")
    saw_binding=False;saw_file=False;seen_binding_keys=set();suffixes=label.source_file_suffixes or ((label.source_file,) if label.source_file else ())
    for role in GROUND_TRUTH_SOURCE_BINDING_ROLES:
        role_rows=[]
        for binding in _evaluation_role_bindings(index,label.contract_address,role):
            key=(binding.source_sha256,binding.artifact_relative_path)
            if key in seen_binding_keys:continue
            seen_binding_keys.add(key);role_rows.append(binding)
        if not role_rows:continue
        saw_binding=True
        if len({item.source_sha256 for item in role_rows})>1:return _repair_failed(label,"GROUND_TRUTH_SOURCE_CACHE_CONFLICT",role)
        files=[]
        for binding in role_rows:
            for normalized,text in _binding_source_files(index,binding):
                files.append((binding.source_sha256,normalized,text))
        unique_files={(sha,name,text):(sha,name,text) for sha,name,text in files}
        files=list(unique_files.values())
        if label.source_file:
            matched=[item for item in files if any(item[1]==suffix or item[1].endswith("/"+suffix) for suffix in suffixes)]
            if not matched and not label.absolute_path_recovered:
                basename=PurePosixPath(label.source_file).name
                matched=[item for item in files if PurePosixPath(item[1]).name==basename]
        else:
            matched=[]
            for item in files:
                try:_utf8_lines(item[2],label.offset,label.length);matched.append(item)
                except ValueError:pass
        dedup={(item[0],item[1],item[2]):item for item in matched};matched=list(dedup.values())
        if not matched:continue
        saw_file=True
        if len(matched)!=1:
            code="GROUND_TRUTH_PATH_SUFFIX_AMBIGUOUS" if label.absolute_path_recovered else "GROUND_TRUTH_FILE_AMBIGUOUS"
            return _repair_failed(label,code,role)
        _,normalized,text=matched[0]
        try:start_line,end_line=_utf8_lines(text,label.offset,label.length)
        except ValueError as exc:return _repair_failed(label,str(exc),role)
        parsed=parse_solidity_source(text);label_end=label.offset+label.length;functions=[]
        for function in parsed.get("functions",[]):
            start=function.get("_start");end=function.get("_end")
            if not isinstance(start,int) or not isinstance(end,int):continue
            start_byte=len(text[:start].encode("utf-8"));end_byte=len(text[:end+1].encode("utf-8"))
            if start_byte<=label.offset and end_byte>=label_end:functions.append(function)
        reasons=list(label.reason_codes);signature=name=None
        if len(functions)==1:
            signature=functions[0].get("canonical_signature") or functions[0].get("function_name");name=functions[0].get("function_name")
        elif len(functions)>1:reasons.append("GROUND_TRUTH_FUNCTION_AMBIGUOUS")
        else:reasons.append("GROUND_TRUTH_FUNCTION_NOT_FOUND")
        same_name=sum(1 for item in parsed.get("functions",[]) if item.get("function_name")==name) if name else 0
        status="RESOLVED_FUNCTION" if signature else "RESOLVED_FILE_AND_LINES"
        return ResolvedLocalizationGroundTruth(status,True,bool(signature),True,start_line,end_line,signature,name,"UTF8_BYTE_OFFSET",tuple(dict.fromkeys(reasons)),normalized,same_name==1,role,label.absolute_path_recovered)
    if not saw_binding:return _repair_failed(label,"GROUND_TRUTH_SOURCE_CACHE_MISSING")
    if label.absolute_path_recovered:
        return _repair_failed(label,"GROUND_TRUTH_PATH_SUFFIX_NO_MATCH")
    if saw_file:return _repair_failed(label,"GROUND_TRUTH_FILE_AMBIGUOUS")
    return _repair_failed(label,"GROUND_TRUTH_FILE_NOT_FOUND")


def _failed(code:str)->ResolvedLocalizationGroundTruth:
    return ResolvedLocalizationGroundTruth("UNRESOLVED",False,False,False,None,None,None,None,"UTF8_BYTE_OFFSET",(code,),None,False)


def _candidate_file(row:Mapping[str,Any])->str|None:
    value=row.get("relative_file") or row.get("file")
    try:return _normal_path(value) if isinstance(value,str) else None
    except ValueError:return None


def evaluate_localization_topk(truth:ResolvedLocalizationGroundTruth,findings:Sequence[Mapping[str,Any]])->LocalizationTopKMetrics:
    ordered=sorted((dict(x) for x in findings if isinstance(x,Mapping)),key=lambda x:int(x.get("rank",999)))[:3]
    if not ordered:return LocalizationTopKMetrics(*(False if value else None for value in (truth.source_file_available,truth.source_file_available,truth.function_available,truth.function_available,truth.line_range_available,truth.line_range_available)),False)
    def hits(k:int,level:str)->bool|None:
        if level=="file" and not truth.source_file_available:return None
        if level=="function" and not truth.function_available:return None
        if level=="line" and not truth.line_range_available:return None
        for row in ordered[:k]:
            file_name=_candidate_file(row)
            file_hit=bool(file_name and truth.normalized_source_file and (file_name==truth.normalized_source_file or file_name.endswith("/"+truth.normalized_source_file) or truth.normalized_source_file.endswith("/"+file_name)))
            if not file_hit:continue
            if level=="file":return True
            if level=="function":
                predicted=row.get("function") or row.get("function_signature")
                if isinstance(predicted,str) and (predicted==truth.function_signature or ("(" not in predicted and truth.function_name_unique_in_file and predicted==truth.function_name)):return True
            if level=="line":
                start=row.get("start_line"); end=row.get("end_line")
                if isinstance(start,int) and isinstance(end,int) and start<=truth.truth_end_line and end>=truth.truth_start_line:return True
        return False
    return LocalizationTopKMetrics(hits(1,"file"),hits(3,"file"),hits(1,"function"),hits(3,"function"),hits(1,"line"),hits(3,"line"),True)


def serialize_localization_ground_truth_summary(value:ResolvedLocalizationGroundTruth|LocalizationTopKMetrics)->str:
    payload=value.public_summary() if isinstance(value,ResolvedLocalizationGroundTruth) else value.to_dict()
    return json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
