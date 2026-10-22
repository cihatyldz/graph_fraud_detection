"""Makaledeki tum sayilar dondurulmus notebook ciktilarindan (ieee.txt, ell.txt, ell_e3.txt, bs.txt) okunur.
Elle sayi yazilmaz: tablolar ve metindeki turetilmis degerler buradan uretilir."""
import re, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TXT = {k: open(os.path.join(HERE, f'{k}.txt')).read().splitlines() for k in ['ieee', 'ell', 'ell_e3', 'bs']}
NUM = re.compile(r'^-?\d+(\.\d+)?$')

def find(src, pat, start=0, nth=0):
    hits = [i for i, l in enumerate(TXT[src]) if pat in l and i >= start]
    assert len(hits) > nth, (src, pat)
    return hits[nth]

def model_table(src, title, cols, start=0, nth=0, skip=4):
    """Baslik satirindan sonra 'model' indeksli basit tablo: ilk token ad, kalan len(cols) token sayi."""
    i = find(src, title, start, nth) + skip
    rows = {}
    for l in TXT[src][i:]:
        t = l.split()
        if not t or l.startswith('==') or l.startswith('D-A') or l.startswith('PR-AUC') or not NUM.match(t[-1]):
            if rows: break
            continue
        nums = [float(x) for x in t[-len(cols):]]
        rows[' '.join(t[:-len(cols)])] = nums
    return pd.DataFrame(rows, index=cols).T


def first_data(src, i):
    """i'den itibaren son tokeni sayi olan ilk satir."""
    while True:
        t = TXT[src][i].split()
        if t and NUM.match(t[-1]) and not TXT[src][i].startswith('=='): return i
        i += 1

SETS4 = ['A', 'B', 'C', 'D']
MODELS10 = ['DT', 'GBDT', 'KNN', 'LR', 'LightGBM', 'MLP', 'NB', 'RF', 'SVM', 'XGBoost']
D = {}

# ------------------------------------------------------------------ IEEE-CIS
ie = {}
ie['p1_f1'] = model_table('ieee', 'PROTOKOL 1 — RASTGELE 5-FOLD CV, F1', SETS4 + ['VestaC', 'C-A', 'C-B', 'D-A'])
ie['p1_pr'] = model_table('ieee', 'PROTOKOL 1 — PR-AUC', SETS4 + ['D-A'])
ie['p2_f1'] = model_table('ieee', 'PROTOKOL 2 — GROUPKFOLD (uid), F1', SETS4 + ['C-A', 'C-B', 'D-A'])
ie['p3_f1'] = model_table('ieee', 'PROTOKOL 3 — ZAMANSAL OUT-OF-TIME, F1', SETS4 + ['C-A', 'C-B', 'D-A'])
ie['p3_pr'] = model_table('ieee', 'PR-AUC:', SETS4, start=find('ieee', 'PROTOKOL 3 — ZAMANSAL'), skip=3)
for k in ['p1_f1', 'p1_pr', 'p2_f1', 'p3_f1', 'p3_pr']:
    assert list(ie[k].index) == MODELS10, (k, list(ie[k].index))
# P2 PR-AUC: C cozum tablosundan degil, B9 ozetinden ortalama (10 model)
def b9(src):
    i = find(src, 'B9 — Anlamli model sayisi'); out = {}
    proto = comp = None
    for l in TXT[src][i + 4:i + 20]:
        m = re.match(r'^(P\d)?\s+(.*?)\s+(f1|pr_auc)\s+(-?\d\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', l)
        if not m: continue
        if m.group(1): proto = m.group(1)
        if m.group(2).strip(): comp = m.group(2).strip()
        out[(proto, comp, m.group(3))] = dict(mean=float(m.group(4)), naive=int(m.group(5)), holm=int(m.group(6)), n=int(m.group(7)))
    return out
ie['b9'] = b9('ieee')

def kappa_line(src, label_pat):
    i = find(src, label_pat); return i

# A8 (guclu baseline, LightGBM)
def a8(src, met):
    i = find(src, f'A8 — guclu baseline ({met})') + 4; out = {}
    for l in TXT[src][i:i + 3]:
        t = l.split(); out[' '.join(t[:-8])] = [float(x) for x in t[-7:]]
    return pd.DataFrame(out, index=['A_full', 'B_full', 'C_full', 'D_full', 'D-A', 'C-B', 'B-A']).T
ie['a8_f1'], ie['a8_pr'] = a8('ieee', 'f1'), a8('ieee', 'pr_auc')

# sizinti enjeksiyonu
def leak(src, met):
    i = find(src, f'SIZINTI ENJEKSIYONU — {met}') + 4; out = {}
    for l in TXT[src][i:i + 7]:
        t = l.split(); out[t[0]] = [float(x) for x in t[1:]]
    return pd.DataFrame(out, index=['LR', 'LightGBM', 'RF', 'mean', 'LR_inf', 'LightGBM_inf', 'RF_inf', 'mean_inf']).T
ie['leak_f1'], ie['leak_pr'] = leak('ieee', 'f1'), leak('ieee', 'pr_auc')

# C cozum katmani: protokol bloklari
C_COLS = ['kappa', 'A', 'C', 'Ck', 'RC', 'dCk_C', 'dCk_A', 'dRC_C', 'D', 'Dk', 'RD', 'dDk_D', 'dDk_A', 'dRD_D']
def csol(src, met):
    i = first_data(src, find(src, f'C — cozum katmani ({met})') + 1); out = {}; proto = None
    for l in TXT[src][i:]:
        if not l.strip() or l.startswith('=='): break
        t = l.split()
        nums = t[-14:]; head = l[:l.index(nums[0])].split()
        if len(head) > 1:
            proto = ' '.join(head[:-1])
        out[(proto, head[-1])] = [float(x) for x in nums]
    return pd.DataFrame(out, index=C_COLS).T
ie['c_f1'], ie['c_pr'] = csol('ieee', 'f1'), csol('ieee', 'pr_auc')

# kappa-ayrilmis (C_P*)
def kstrat(src, title, models):
    i = first_data(src, find(src, title) + 1); out = {}; tag = sub = None
    for l in TXT[src][i:]:
        if not l.strip() or l.startswith('=='): break
        t = l.split(); nums = t[-(len(models) + 1):]
        head = l[:l.rindex(nums[0], 0, len(l) - len(' '.join(nums[1:])) - 1)].rstrip() if False else None
        name_part = l.rsplit(None, len(models) + 1)[0]
        toks = name_part.split()
        if toks[0].startswith(('C_P', 'E2_P', 'E4_P', 'R_P', 'E3_')):
            tag = toks[0]; toks = toks[1:]
        if toks[0] in ('gorulen', 'gorulmeyen'):
            sub = toks[0]; toks = toks[1:]
        out[(tag, sub, ' '.join(toks))] = [float(x) for x in nums]
    return pd.DataFrame(out, index=models + ['mean']).T
ie['kpr'] = kstrat('ieee', 'C — kappa-ayrilmis pr_auc', ['LR', 'LightGBM', 'MLP', 'RF', 'XGBoost'])

def counts(src, title, start=0):
    i = find(src, title, start) + 1
    while not TXT[src][i].strip().startswith('fold'): i += 1
    i += 2; out = {}; tag = None
    for l in TXT[src][i:]:
        t = l.split()
        if not t or not NUM.match(t[-2]): break
        if len(t) == 8: tag = t[0]; t = t[1:]
        out[(tag, t[0])] = dict(fold=int(t[1]), n=int(t[2]), pos=int(t[3]), minpos=int(t[4]), prev=float(t[5]), ok=t[6])
    return out
ie['counts'] = counts('ieee', 'C — kappa alt kumeleri: boyut')

# P4
p4i = find('ieee', 'P4 origin\'leri:') + 2
ie['p4_origins'] = [l.split() for l in TXT['ieee'][p4i:p4i + 5]]
def p4mean(met):
    i = find('ieee', f'P4 — {met}, origin ortalamasi') + 4; out = {}
    for l in TXT['ieee'][i:i + 14]:
        t = l.rsplit(None, 2); out[t[0].strip()] = [float(t[1]), float(t[2])]
    return pd.DataFrame(out, index=['LightGBM', 'XGBoost']).T
ie['p4_pr'], ie['p4_f1'] = p4mean('pr_auc'), p4mean('f1')
def p4diff(met):
    i = find('ieee', f'P4 — farklar ({met})') + 3; out = {}; comp = None
    for l in TXT['ieee'][i:i + 22]:
        t = l.split()
        if not t: break
        win, sd, mn, mdl = t[-1], float(t[-2]), float(t[-3]), t[-4]
        if len(t) > 4: comp = ' '.join(t[:-4])
        out[(comp, mdl)] = dict(mean=mn, sd=sd, wins=win)
    return out
ie['p4_dpr'], ie['p4_df1'] = p4diff('pr_auc'), p4diff('f1')
i = find('ieee', 'P4 — origin bazinda PR-AUC farklari') + 4
ie['p4_origin_pr'] = [l.split() for l in TXT['ieee'][i:i + 5]]
ie['p4_kappa'] = [float(r[2]) for r in ie['p4_origin_pr']]
ie['p4_counts'] = counts('ieee', 'P4 — kappa alt kumeleri')
def p4k():
    i = find('ieee', 'P4 — kappa-ayrilmis PR-AUC') + 3; out = {}; sub = None
    for l in TXT['ieee'][i:i + 14]:
        t = l.rsplit(None, 3); name = t[0].split()
        if name[0] in ('gorulen', 'gorulmeyen'): sub = name[0]; name = name[1:]
        out[(sub, ' '.join(name))] = [float(x) for x in t[1:]]
    return pd.DataFrame(out, index=['LightGBM', 'XGBoost', 'mean']).T
ie['p4_k'] = p4k()

# denetim, B10, SHAP, maliyet
def audit(src):
    i = find(src, 'Kriter Kabul') + 1; out = {}
    for l in TXT[src][i:i + 5]:
        m = re.search(r'(C\d).*?([<>]) ([\d.]+) ([\d.]+)\s+(!! KALDI|GECTI)', l)
        out[m.group(1)] = dict(val=float(m.group(4)), passed=m.group(5) == 'GECTI')
    return out
ie['audit'] = audit('ieee')
def samp(src):
    i = find(src, 'SINIF DENGESİZLİĞİ STRATEJİLERİ') + 4; out = {}
    for l in TXT[src][i:i + 5]:
        t = l.split(); out[t[0]] = [float(x) for x in t[1:]]
    return pd.DataFrame(out, index=['acc', 'prec', 'rec', 'f1', 'auc', 'pr', 'mcc', 'fpr']).T
ie['samp'] = samp('ieee')
def shap(src):
    i = find(src, 'KATEGORİ BAZINDA SHAP KATKI PAYI') + 4; out = {}
    for l in TXT[src][i:i + 3]:
        t = l.split(); out[t[0]] = float(t[-1])
    return out
ie['shap'] = shap('ieee')
i = find('ieee', 'Algoritma bazlı ortalama süre'); ie['alg_time'] = {l.split()[0]: float(l.split()[1]) for l in TXT['ieee'][i + 1:i + 6]}
i = find('ieee', 'EŞİK SEÇİMİNİN ETKİSİ') + 3
ie['total_extract'] = float(re.search(r'([\d.]+) s', TXT['ieee'][find('ieee', 'Toplam çıkarım süresi')]).group(1))
ie['thr'] = {l.split()[0]: (float(l.split()[1]), float(l.split()[2])) for l in TXT['ieee'][i:i + 5]}
ie['thr_value'] = float(re.search(r'eşik: ([\d.]+)', TXT['ieee'][find('ieee', 'Eğitim fold\'unda seçilen eşik')]).group(1))
ie['fp'] = TXT['ieee'][find('ieee', 'Orneklem parmak izi')]
D['ieee'] = ie

# ------------------------------------------------------------------ Elliptic++
el = {}
E_COLS = ['kappa', 'A', 'B', 'C', 'D', 'AA', 'Ck', 'Dk', 'RC', 'RD', 'D-A', 'C-B', 'Ck-C', 'RC-A', 'RC-C']
def etab(title, cols, src='ell'):
    i = first_data(src, find(src, title) + 1); out = {}; proto = None
    for l in TXT[src][i:]:
        if not l.strip() or l.startswith('=='): break
        t = l.split(); nums = t[-len(cols):]
        head = l.rsplit(None, len(cols))[0].split()
        if len(head) > 1: proto = ' '.join(head[:-1])
        out[(proto, head[-1])] = [float(x) for x in nums]
    return pd.DataFrame(out, index=cols).T
el['e2_f1'], el['e2_pr'] = etab('E2 — f1', E_COLS), etab('E2 — pr_auc', E_COLS)
E4_COLS = ['kappa', 'AA', 'AA+B', 'AA+C', 'AA+D', 'AA+Ck', 'AA+Dk', 'RAC', 'RAD', 'B-AA', 'D-AA', 'C-AA', 'C-B', 'Ck-C', 'RC-AA', 'RD-AA']
el['e4_pr'], el['e4_f1'] = etab('E4 — pr_auc (AA = A+AGG)', E4_COLS), etab('E4 — f1 (AA = A+AGG)', E4_COLS)
el['audit'] = audit('ell')
el['kpr'] = kstrat('ell', 'E2 — kappa-ayrilmis pr_auc', ['LR', 'LightGBM', 'RF', 'XGBoost'])
def k4(title):
    i = first_data('ell', find('ell', title) + 1) if False else find('ell', title) + 4; out = {}; tag = sub = None
    for l in TXT['ell'][i:]:
        t = l.split()
        if not t or l.startswith('=='): break
        if t[-1] not in ('evet', 'HAYIR'): continue
        nums = [float(x) for x in t[-6:-1]]; toks = t[:-6]
        if toks[0].startswith('E4_P'): tag = toks[0]; toks = toks[1:]
        if toks[0] in ('gorulen', 'gorulmeyen'): sub = toks[0]; toks = toks[1:]
        out[(tag, sub, ' '.join(toks))] = nums
    return pd.DataFrame(out, index=['LR', 'LightGBM', 'RF', 'XGBoost', 'mean']).T
el['k4pr'] = k4('E4 — kappa-ayrilmis pr_auc')
i = find('ell', 'Alt grup boyutlari (fold ortalamasi)') + 3
el['sizes'] = {}
tag = None
for l in TXT['ell'][i:i + 6]:
    t = l.split()
    if len(t) == 5: tag = t[0]; t = t[1:]
    el['sizes'][(tag, t[0])] = dict(n=float(t[1]), pos=float(t[2]), prev=float(t[3]))
el['e4_counts'] = counts('ell', 'Alt kume boyutlari (KAPPA_MIN_POS=30)')
def b9e(title):
    i = find('ell', title); out = {}; comp = None
    for l in TXT['ell'][i:i + 14]:
        m = re.match(r'^(.*?)\s+(f1|pr_auc)\s+(-?\d\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', l)
        if not m: continue
        if m.group(1).strip(): comp = m.group(1).strip()
        out[(comp, m.group(2))] = dict(mean=float(m.group(3)), naive=int(m.group(4)), holm=int(m.group(5)), n=int(m.group(6)))
    return out
el['b9_e2'] = b9e('ort_fark  anlamli_naive  anlamli_holm  model')
el['b9_e4'] = b9e('A+AGG+B vs A+AGG (referans)')
# E4 P4 kazanan origin
def wins(title):
    i = find('ell', title) + 3; out = {}; comp = None
    for l in TXT['ell'][i:i + 28]:
        t = l.split()
        if len(t) < 4: break
        if len(t) == 5: comp = t[0]; t = t[1:]
        out[(comp, t[0])] = dict(mean=float(t[1]), sd=float(t[2]), wins=t[3])
    return out
el['e4_wins_pr'] = wins('Model bazinda (pr_auc): ortalama ± std, kazanan origin')
el['e4_wins_f1'] = wins('Model bazinda (f1): ortalama ± std, kazanan origin')
def origin_rows(title, cols):
    i = find('ell', title) + 4; out = {}
    for l in TXT['ell'][i:i + 5]:
        t = l.split(); out[f'{t[0]} {t[1]}'] = [float(x) for x in t[2:]]
    return pd.DataFrame(out, index=cols).T
el['e2_origin_pr'] = origin_rows('Rolling-origin — pr_auc (modellerin ortalamasi)', ['kappa', 'npos', 'A', 'C', 'D', 'Ck', 'RC', 'AA', 'D-A', 'RC-A', 'Ck-C'])
el['e4_origin_pr'] = origin_rows('E4 rolling-origin — pr_auc (modellerin ortalamasi)',
                                 ['kappa', 'npos', 'AA', 'AA+C', 'AA+D', 'AA+Ck', 'RAC', 'RAD', 'B-AA', 'D-AA', 'C-AA', 'C-B', 'Ck-C', 'RC-AA', 'RD-AA'])
# GNN
def gnn():
    i = find('ell_e3', 'E3 — GNN (tohum') + 4; out = {}; proto = None
    for l in TXT['ell_e3'][i:i + 12]:
        nums = [float(x) for x in l.split()[-8:]]
        head = l.rsplit(None, 8)[0].strip()
        for pl in ['Random 5-fold CV', 'Rolling-origin', 'Temporal 1-34 / 35-49']:
            if head.startswith(pl): proto, head = pl, head[len(pl):].strip()
        out[(proto, head)] = nums
    return pd.DataFrame(out, index=['f1', 'pr', 'auc', 'train_s', 'infer_s', 'ep', 'f1_sd', 'pr_sd']).T
el['gnn'] = gnn()
i = find('ell', 'GNN — kappa-ayrilmis PR-AUC') + 3
el['gnn_k'] = [l.split() for l in TXT['ell'][i:i + 6]]
i = find('ell_e3', 'ms_per_1k') + 1
el['lat'] = [l.rsplit(None, 3) for l in TXT['ell_e3'][i:i + 6]]
D['ell'] = el

# ------------------------------------------------------------------ BankSim
bs = {}
bs['p1_f1'] = model_table('bs', 'PROTOKOL 1 — RASTGELE 5-FOLD CV (SIZINTISIZ), F1', SETS4 + ['C-A', 'C-B', 'D-A'])
bs['p1_pr'] = model_table('bs', 'PROTOKOL 1 — PR-AUC', SETS4 + ['C-A', 'D-A'])
bs['p2_f1'] = model_table('bs', 'PROTOKOL 2 — GROUPKFOLD (müşteri bazlı), F1', SETS4 + ['C-A', 'C-B', 'D-A'])
bs['p3_f1'] = model_table('bs', 'PROTOKOL 3 — ZAMANSAL OUT-OF-TIME, F1', SETS4 + ['C-A', 'C-B', 'D-A'])
bs['p3_pr'] = model_table('bs', 'PR-AUC:', SETS4, start=find('bs', 'PROTOKOL 3 — ZAMANSAL'), skip=3)
for k in ['p1_f1', 'p1_pr', 'p2_f1', 'p3_f1', 'p3_pr']:
    assert list(bs[k].index) == MODELS10, (k, list(bs[k].index))
bs['b9'] = b9('bs')
bs['leak_f1'], bs['leak_pr'] = leak('bs', 'f1'), leak('bs', 'pr_auc')
bs['c_f1'], bs['c_pr'] = csol('bs', 'f1'), csol('bs', 'pr_auc')
bs['audit'] = audit('bs')
bs['samp'] = samp('bs')
bs['shap'] = shap('bs')
D['bs'] = bs

# ------------------------------------------------------------------ turetilmis ozet: protokol x veri seti
def mean_diff(tab, a='D', b='A'): return float((tab[a] - tab[b]).mean())
S = {}
S['ieee'] = {
    'P1': dict(kappa=0.759, f1=mean_diff(ie['p1_f1']), pr=mean_diff(ie['p1_pr']), nm=10),
    'P2': dict(kappa=0.0, f1=mean_diff(ie['p2_f1']), pr=ie['b9'][('P2', 'D (A+graf) vs A (ham)', 'pr_auc')]['mean'], nm=10),
    'P3': dict(kappa=0.441, f1=mean_diff(ie['p3_f1']), pr=mean_diff(ie['p3_pr']), nm=10),
    'P4': dict(kappa=float(np.mean(ie['p4_kappa'])), f1=float(np.mean([ie['p4_df1'][('D−A', m)]['mean'] for m in ['LightGBM', 'XGBoost']])),
               pr=float(np.mean([ie['p4_dpr'][('D−A', m)]['mean'] for m in ['LightGBM', 'XGBoost']])), nm=2),
}
S['ell'] = {p: dict(kappa=float(el['e2_f1'].loc[(lab, 'ORTALAMA'), 'kappa']), f1=float(el['e2_f1'].loc[(lab, 'ORTALAMA'), 'D-A']),
                    pr=float(el['e2_pr'].loc[(lab, 'ORTALAMA'), 'D-A']), nm=4)
            for p, lab in [('P1', 'Random 5-fold CV'), ('P3', 'Temporal 1-34 / 35-49'), ('P4', 'Rolling-origin')]}
S['bs'] = {
    'P1': dict(kappa=1.0, f1=mean_diff(bs['p1_f1']), pr=mean_diff(bs['p1_pr']), nm=10),
    'P2': dict(kappa=0.0, f1=mean_diff(bs['p2_f1']), pr=bs['b9'][('P2', 'D (A+graf) vs A (ham)', 'pr_auc')]['mean'], nm=10),
    'P3': dict(kappa=1.0, f1=mean_diff(bs['p3_f1']), pr=mean_diff(bs['p3_pr']), nm=10),
}
D['summary'] = S

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for ds, v in S.items():
        print(ds, {p: {k: round(x, 4) if isinstance(x, float) else x for k, x in d.items()} for p, d in v.items()})
    print(ie['c_pr'].round(4).head(18))
    print(el['gnn'])
    print(el['e4_wins_pr'])
    print(ie['p4_counts'], el['e4_counts'])
    print(ie['audit'], el['audit'], bs['audit'])
