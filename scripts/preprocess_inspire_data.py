"""Feature registry and compatibility CLI; unsafe operation-only preprocessing retired."""
LABS = ['albumin','alp','alt','aptt','ast','be','bun','calcium','chloride','ck','ckmb','creatinine','crp','d_dimer','fibrinogen','glucose','hb','hba1c','hco3','hct','ica','lactate','lymphocyte','paco2','pao2','ph','phosphorus','platelet','potassium','ptinr','sao2','seg','sodium','total_bilirubin','total_protein','troponin_i','troponin_t','wbc']
VITALS = ['aft','air','alb20','alb5','art_dbp','art_mbp','art_sbp','bis','bt','cbro2','ci','cpat','cryo','cvp','d10w','d50w','d5w','ds','dobui','dopai','ebl','eph','epi','epii','etco2','etdes','etgas','etiso','etsevo','ffp','fio2','ftn','hes','hns','hr','hs','mdz','minvol','mlni','n2o','nepi','nibp_dbp','nibp_mbp','nibp_sbp','ns','ntgi','o2','pap_dbp','pap_mbp','pap_sbp','pc','peep','pepi','phe','pheresis','pip','pmean','ppf','ppfi','pplat','psa','rbc','rfti','rr','sft','spo2','sti','stii','stiii','stv5','svi','uo','vaso','vt']
WARD = ['art_sbp','bt','crrt','ecmo','fio2','gcs_e','gcs_m','gcs_v','hr','iabp','nibp_dbp','nibp_mbp','nibp_sbp','rr','spo2','uo','vent']
FEATURES = [(t,n) for t,ns in [('labs',LABS),('vitals',VITALS),('ward_vitals',WARD)] for n in ns]
FIDX = {f:i for i,f in enumerate(FEATURES)}


def main():
    from hcmt.data.preprocess_timeline import main as corrected_main
    corrected_main()

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
