"""Bounded-memory INSPIRE aggregation. All timestamps are source MINUTES."""
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.preprocess_inspire_data import FEATURES, FIDX


# INSPIRE's raw laboratory export contains this historical spelling.  Keep the
# canonical feature name downstream while making the source correction
# explicit and auditable instead of silently dropping lactate measurements.
ITEM_NAME_ALIASES = {
    ("labs", "lacate"): "lactate",
}


def stream(path, chunksize=250000):
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(['gzip','-dc',str(path)], stdout=subprocess.PIPE, stderr=err)
        try:
            yield from pd.read_csv(p.stdout, chunksize=chunksize, low_memory=False)
        finally:
            p.stdout.close()
            status = p.wait()
            err.seek(0)
            message = err.read().decode(errors='replace')
            if status and not (status == 2 and 'trailing garbage ignored' in message):
                raise RuntimeError(f'Cannot read {path}: exit {status}: {message}')


def build_admissions(ops):
    ops = ops.copy()
    for c in ('subject_id','hadm_id','op_id','admission_time','discharge_time',
              'inhosp_death_time','icuin_time','orin_time'):
        ops[c] = pd.to_numeric(ops[c], errors='coerce')
    required = ['subject_id','hadm_id','op_id','admission_time','discharge_time']
    if ops[required].isna().any().any() or (ops.discharge_time <= ops.admission_time).any():
        raise ValueError('Invalid admission interval/IDs; refusing invented timestamps.')
    for c in ('subject_id','hadm_id','op_id'):
        ops[c] = ops[c].astype('int64')
    if ops.op_id.duplicated().any():
        raise ValueError('Duplicate op_id')
    rows = []
    for (s,h), g in ops.groupby(['subject_id','hadm_id'], sort=True):
        if g.admission_time.nunique()!=1 or g.discharge_time.nunique()!=1:
            raise ValueError(f'Conflicting admission interval for hadm_id {h}')
        deaths = g.inhosp_death_time.dropna().unique()
        if len(deaths)>1: raise ValueError('Conflicting death timestamps')
        start,end = float(g.admission_time.iloc[0]),float(g.discharge_time.iloc[0])
        death = float(deaths[0]) if len(deaths) else np.nan
        if np.isfinite(death) and death <= start:
            raise ValueError('Death precedes admission; cannot define at-risk anchors')
        # INSPIRE may repeat a later in-hospital death on an earlier operation.
        # Preserve the timestamp for audit; labels only count death within THIS
        # admission and censor at its discharge (never extend follow-up).
        first = g.sort_values('orin_time').iloc[0]
        rows.append(dict(subject_id=s,hadm_id=h,start=start,end=end,death=death,
                         icu_times=json.dumps(sorted(set(g.icuin_time.dropna().astype(float)))),
                         static_known_time=float(first.orin_time), age=first.get('age',np.nan),
                         sex=first.get('sex',''),weight=first.get('weight',np.nan),height=first.get('height',np.nan)))
    adm=pd.DataFrame(rows).sort_values(['subject_id','start','hadm_id']).reset_index(drop=True)
    adm['adm_idx']=np.arange(len(adm));adm['length']=np.floor((adm.end-adm.start)/5).astype('int64')+1
    exact=adm.set_index(['subject_id','hadm_id']).adm_idx.to_dict()
    return adm,{int(r.op_id):exact[(r.subject_id,r.hadm_id)] for r in ops.itertuples()}


def assign_records(frame, adm, subjects, opmap, table):
    out=np.full(len(frame),-1,dtype='int64')
    times=pd.to_numeric(frame.chart_time,errors='coerce').to_numpy(float)
    if table=='vitals':
        mapped=pd.to_numeric(frame.op_id,errors='coerce').map(opmap).fillna(-1).to_numpy('int64')
        safe=np.maximum(mapped,0)
        good=(mapped>=0)&(times>=adm.start.to_numpy()[safe])&(times<=adm.end.to_numpy()[safe])
        if 'subject_id' in frame:
            good &= pd.to_numeric(frame.subject_id,errors='coerce').to_numpy()==adm.subject_id.to_numpy()[safe]
        out[good]=mapped[good]
    else:
        sc=pd.to_numeric(frame.subject_id,errors='coerce')
        for s,ix in sc.groupby(sc).groups.items():
            entries=subjects.get(s)
            if entries is None: continue
            ix=np.asarray(ix);count=np.zeros(len(ix),dtype=int)
            for start,end,ai in entries:
                match=(times[ix]>=start)&(times[ix]<=end)
                out[ix[match]]=ai;count+=match
            out[ix[count>1]]=-2
    out[~np.isfinite(times)]=-3
    return out,times


def preprocess(source,output,chunksize=250000,test_mode=False):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError(f'Output must be empty; preserving {output}')
    ops=pd.concat(stream(source/'operations.csv.gz',chunksize),ignore_index=True)
    if test_mode: ops=ops[ops.subject_id.isin(ops.subject_id.drop_duplicates().head(20))].copy()
    adm,opmap=build_admissions(ops)
    print(f'admissions={len(adm):,}, operations={len(ops):,}, five_minute_bins={adm.length.sum():,}',flush=True)
    subjects={s:list(g[['start','end','adm_idx']].itertuples(index=False,name=None)) for s,g in adm.groupby('subject_id')}
    db=sqlite3.connect(output/'aggregation.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE obs(ai INTEGER,b INTEGER,f INTEGER,t REAL,v REAL,n INTEGER,PRIMARY KEY(ai,b,f)) WITHOUT ROWID')
    db.execute('CREATE TABLE codes(ai INTEGER,b INTEGER,code INTEGER,PRIMARY KEY(ai,b,code)) WITHOUT ROWID')
    vocab={'<PAD>':0,'<UNK>':1};audit={'operations':{'rows':len(ops),'assigned':len(ops),
        'death_after_this_discharge':int((adm.death>adm.end).sum())},'test_mode':test_mode}
    phase_records=[]
    for r in ops.itertuples():
        ai=opmap[int(r.op_id)];ad=adm.iloc[ai]
        for col in ('orin_time','orout_time','opstart_time','opend_time','anstart_time','anend_time',
                    'cpbon_time','cpboff_time','icuin_time','icuout_time'):
            t=pd.to_numeric(getattr(r,col,np.nan),errors='coerce')
            if pd.isna(t) or not ad.start<=t<=ad.end:continue
            code='phase:'+col
            if code not in vocab:vocab[code]=len(vocab)
            b=int(np.ceil((t-ad.start)/5));phase_records.append((ai,b,vocab[code]))
        # Conservative availability: surgery attributes only at OR entry, never
        # future operation type/ASA projected backward to admission.
        t=pd.to_numeric(getattr(r,'orin_time',np.nan),errors='coerce')
        if pd.notna(t) and ad.start<=t<=ad.end:
            for col in ('department','asa','antype'):
                value=getattr(r,col,np.nan)
                if pd.isna(value):continue
                code=f'{col}:{value}'
                if code not in vocab:vocab[code]=len(vocab)
                phase_records.append((ai,int(np.ceil((t-ad.start)/5)),vocab[code]))
    db.executemany('INSERT OR IGNORE INTO codes VALUES(?,?,?)',phase_records);db.commit()
    sql='''INSERT INTO obs VALUES(?,?,?,?,?,?) ON CONFLICT(ai,b,f) DO UPDATE SET
      v=CASE WHEN excluded.t>obs.t THEN excluded.v WHEN excluded.t=obs.t THEN obs.v+excluded.v ELSE obs.v END,
      n=CASE WHEN excluded.t>obs.t THEN excluded.n WHEN excluded.t=obs.t THEN obs.n+excluded.n ELSE obs.n END,
      t=MAX(obs.t,excluded.t)'''
    for table in ('vitals','labs','ward_vitals','medications','diagnosis'):
        stats=dict(rows=0,assigned=0,outside_admission=0,ambiguous=0,invalid_time=0,invalid_value=0,unknown_feature=0)
        for chunk in stream(source/(table+'.csv.gz'),chunksize):
            chunk=chunk.reset_index(drop=True)
            assignment,times=assign_records(chunk,adm,subjects,opmap,table)
            stats['rows']+=len(chunk)
            for name,code in [('outside_admission',-1),('ambiguous',-2),('invalid_time',-3)]:stats[name]+=int((assignment==code).sum())
            good=assignment>=0;stats['assigned']+=int(good.sum())
            c=chunk[good].copy();c['ai']=assignment[good];c['t']=times[good]
            # Right-closed bin: no observation becomes available before its time.
            c['b']=np.ceil((c.t.to_numpy()-adm.start.to_numpy()[assignment[good]])/5).astype('int64')
            if table in ('vitals','labs','ward_vitals'):
                c['f']=[FIDX.get((table, ITEM_NAME_ALIASES.get((table, str(x)), str(x))), -1)
                        for x in c.item_name]
                c['v']=pd.to_numeric(c.value,errors='coerce')
                stats['unknown_feature']+=int((c.f<0).sum());stats['invalid_value']+=int((~np.isfinite(c.v)).sum())
                c=c[(c.f>=0)&np.isfinite(c.v)]
                g=c.groupby(['ai','b','f','t']).v.agg(['sum','count']).reset_index()
                db.executemany(sql,g.itertuples(index=False,name=None))
            else:
                cols=[x for x in (['icd10_cm'] if table=='diagnosis' else ['drug_name','drug_name2','drug_name3','atc_code','atc_code2','atc_code3','route']) if x in c]
                records=[]
                for col in cols:
                    namespace='drug' if col.startswith('drug_name') else 'atc' if col.startswith('atc_code') else col
                    for ai,b,value in c[['ai','b',col]].itertuples(index=False,name=None):
                        if pd.isna(value) or not str(value).strip():continue
                        code=f'{namespace}:{str(value).strip()}'
                        if code not in vocab:vocab[code]=len(vocab)
                        records.append((int(ai),int(b),vocab[code]))
                db.executemany('INSERT OR IGNORE INTO codes VALUES(?,?,?)',records)
            db.commit()
            print(f'{table}: read={stats["rows"]:,}, assigned={stats["assigned"]:,}',flush=True)
        audit[table]=stats;(output/'source_audit.json').write_text(json.dumps(audit,indent=2))
    db.execute('CREATE INDEX obs_feature_idx ON obs(f)');db.commit()
    center,scale=[],[]
    for f in range(len(FEATURES)):
        vals=np.fromiter((r[0] for r in db.execute('SELECT v/n FROM obs WHERE f=?',(f,))),dtype='float64')
        center.append(float(np.median(vals)) if len(vals) else 0.)
        spread=float(np.quantile(vals,.75)-np.quantile(vals,.25)) if len(vals) else 1.
        scale.append(spread if spread>1e-6 else max(float(np.std(vals)),1.) if len(vals) else 1.)
    (output/'normalizer.json').write_text(json.dumps(dict(center=center,scale=scale,method='median_IQR',fit_split='train')))
    for table,names,dtypes,query in [
        ('obs',['obs_bin','obs_feature','obs_value'],['int32','int16','float32'],'SELECT ai,b,f,v/n FROM obs ORDER BY ai,b,f'),
        ('codes',['code_bin','code_id'],['int32','int32'],'SELECT ai,b,code FROM codes ORDER BY ai,b,code')]:
        count=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
        arrays=[np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=dtype,shape=(count,)) for name,dtype in zip(names,dtypes)]
        ptr=np.zeros(len(adm)+1,dtype='int64');cur=db.execute(query);start=0
        while True:
            rows=cur.fetchmany(chunksize)
            if not rows:break
            arr=np.asarray(rows);np.add.at(ptr,arr[:,0].astype(int)+1,1)
            for k,mm in enumerate(arrays):mm[start:start+len(rows)]=arr[:,k+1]
            start+=len(rows)
        for mm in arrays:mm.flush()
        np.save(output/('obs_admission_ptr.npy' if table=='obs' else 'code_admission_ptr.npy'),np.cumsum(ptr))
    static=np.zeros((len(adm),20),dtype='float32')
    static[:,0]=pd.to_numeric(adm.age,errors='coerce').fillna(0).to_numpy()/100;static[:,1]=adm.sex.eq('M').to_numpy()
    static[:,2]=pd.to_numeric(adm.weight,errors='coerce').fillna(0).to_numpy()/110
    static[:,3]=pd.to_numeric(adm.height,errors='coerce').fillna(0).to_numpy()/200
    static[:,4]=adm.weight.notna()&adm.height.notna();np.save(output/'static.npy',static)
    adm.drop(columns=['age','sex','weight','height']).to_csv(output/'admissions.csv',index=False)
    (output/'vocabulary.json').write_text(json.dumps(vocab,ensure_ascii=False));db.close()
    meta=dict(version=6,complete=True,source_time_unit='minutes',bin_minutes=5,storage='sparse_observations',
              num_admissions=len(adm),num_features=len(FEATURES),num_codes=len(vocab),num_static=20,total_bins=int(adm.length.sum()),
              features=[f'{t}:{n}' for t,n in FEATURES],forward_fill_minutes=[60 if t=='labs' else 15 for t,_ in FEATURES],
              split='all_train',event_definition='future in-hospital death or recorded postoperative ICU transfer',
              allcause_death='not used: registry follow-up cutoff unavailable',source_audit='source_audit.json',test_mode=test_mode)
    (output/'admission_timeline_meta.json').write_text(json.dumps(meta,indent=2))
    print('v6 preprocessing complete; old outputs untouched.',flush=True)
    return meta


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--chunksize',type=int,default=250000);p.add_argument('--test_mode',action='store_true')
    args=p.parse_args();preprocess(args.input,args.output,args.chunksize,args.test_mode)
