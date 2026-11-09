"""Transparent descriptive statistics over a user-exported browser session.

This is an exploratory audit: latency is browser monotonic timing, not acoustic
mouth-to-ear latency. Text is never needed for the summary calculations.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone


def percentile(values, fraction: float):
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid percentile')
    ordered = sorted(values)
    point = (len(ordered)-1)*fraction
    lower = int(point)
    frac = point - lower
    return round(ordered[lower] * (1-frac) + ordered[min(lower+1,len(ordered)-1)]*frac, 2)


def _metric(values):
    return {'n':len(values),'p50':percentile(values,.5),'p95':percentile(values,.95),
            'p99':percentile(values,.99),'max':max(values) if values else None}


def summarise(session: dict):
    if session.get('schema_version') != '0.4-live-client':
        raise ValueError('Not an OpenHear M2-alpha session export')
    rows = session.get('audit_records', session.get('turns', []))
    if not isinstance(rows,list) or len(rows)>500:
        raise ValueError('Invalid or excessive audit-record count')
    groups = defaultdict(list)
    for index, record in enumerate(rows):
        if not isinstance(record,dict) or record.get('source') not in ('en','zh') or record.get('target') not in ('en','zh'):
            raise ValueError(f'Invalid record {index}')
        if record['source']==record['target']:
            raise ValueError(f'Invalid same-language record {index}')
        groups[f"{record['source']}-{record['target']}"].append(record)
    summary={}
    for direction in ('en-zh','zh-en'):
        records=groups[direction]
        completed=[r for r in records if r.get('status') in ('completed','completed_text_only')]
        def values(key):
            items=[]
            for r in records:
                v=r.get(key)
                if type(v) in (int,float) and 0<=v<=3600000:
                    items.append(float(v))
            return items
        summary[direction]={'recorded':len(records),
            'completed':len(completed), 'failed':len(records)-len(completed),
            'voice_generated':sum(r.get('tts_status')=='generated' for r in records),
            'pause_to_result_ms':_metric(values('pause_to_result_ms')),
            'browser_request_ms':_metric(values('request_ms'))}
    dropped=session.get('dropped',0);overflow=session.get('audit_overflow',0)
    if type(dropped) is not int or dropped <0 or type(overflow) is not int or overflow<0:
        raise ValueError('Invalid counter values')
    total=len(rows)+dropped
    return {'schema_version':'0.4-live-summary',
            'utc':datetime.now(timezone.utc).isoformat(),
            'total_audit_records':len(rows),'dropped':dropped,
            'estimated_drop_fraction':round(dropped/total,6) if total else None,
            'audit_overflow':overflow,
            'evidence_complete_for_all_turns':overflow==0,
            'by_direction':summary,
            'warning':'Descriptive browser timings only; last-turn text is not evaluated. Not acoustic mouth-to-ear latency; no independent network-isolation evidence.'}


def report_file(source: str, destination: str):
    p=Path(source)
    if p.stat().st_size>8*1024*1024:
        raise ValueError('Session export exceeds 8 MiB')
    session=json.loads(p.read_text(encoding='utf-8'))
    summary=summarise(session)
    from .offline import _write_exclusive
    return _write_exclusive(destination,summary),summary
