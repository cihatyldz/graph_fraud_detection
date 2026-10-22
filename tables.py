"""LaTeX tablolari dogrudan safeg_numbers'tan uretilir (elle sayi girisi yok)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import safeg_numbers as N

D, S = N.D, N.D['summary']
ie, el, bs = D['ieee'], D['ell'], D['bs']

def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '--'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    return s.replace('-', '$-$') if sign or x < 0 else s

def table(caption, label, colspec, header, rows, notes=None, wide=False, size='\\footnotesize', pos='t'):
    """Duz tabular; yalnizca metin genisligini asarsa olceklenir (adjustbox max width)."""
    env = 'table*'   # sn-jnl: table ortami threeparttable; adjustbox yalnizca table* ile uyumlu
    out = [f'\\begin{{{env}}}[{pos}]', size, '\\setlength{\\tabcolsep}{4pt}', f'\\caption{{{caption}}}\\label{{{label}}}',
           '\\begin{adjustbox}{max width=\\textwidth}', f'\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}', '\\toprule']
    out += header + ['\\midrule'] + rows + ['\\botrule', '\\end{tabular}', '\\end{adjustbox}']
    if notes: out.append(f'\\footnotetext{{{notes}}}')
    out.append(f'\\end{{{env}}}')
    return '\n'.join(out)

T = {}

# ---------------------------------------------------------------- audit
def mark(c): return '' if c['passed'] else '$^{\\dagger}$'
rows = []
crit = [('C1', 'Best single-feature AUC $a^*$', '$<0.90$', 'shortcut separability'),
        ('C2', 'Best one/two-condition rule F1 $F_1^*$', '$<0.50$', 'shortcut separability'),
        ('C3', 'Single-transaction entity share $\\rho_1$', '$<0.50$', 'entity-history degeneracy'),
        ('C4', 'Edge-to-node ratio $\\lambda$', '$>1.5$', 'graph degeneracy'),
        ('C5', 'Largest-component share $\\gamma$', '$>0.80$', 'graph fragmentation')]
for c, name, thr, what in crit:
    vals = [f"{f(d[c]['val'], 3 if d[c]['val'] < 10 else 2)}{mark(d[c])}" for d in (ie['audit'], el['audit'], bs['audit'])]
    rows.append(f'{c} & {name} & {thr} & ' + ' & '.join(vals) + f' & {what} \\\\')
T['audit'] = table('Pre-registered artefact audit (diagnostic profile). Criteria and thresholds were fixed before any model was trained; '
                   'they are reported as a profile that constrains interpretation, not as a gate. $^{\\dagger}$\\,criterion not met. '
                   'Values are computed on the training portion of the standard temporal split.',
                   'tab:audit', 'llccccl',
                   ['& Criterion & Accept if & IEEE-CIS & Elliptic++ & BankSim & Diagnoses \\\\'], rows, wide=True)

# ---------------------------------------------------------------- leakage injection
lab = {'+graph': 'Graph and centralities on all data', '+agg': 'Behavioural aggregates on all data',
       '+cfr': 'Community fraud rate in-sample on all data', '+thr': 'Threshold tuned on the test fold',
       'graph+agg+cfr': 'Graph + aggregates + community rate', 'hepsi': 'All four paths'}
rows = []
for k, name in lab.items():
    rows.append(f'{name} & ' + ' & '.join([f(ie['leak_f1'].loc[k, 'mean_inf'], sign=True), f(ie['leak_pr'].loc[k, 'mean_inf'], sign=True),
                                         f(ie['leak_f1'].loc[k, 'LightGBM_inf'], sign=True),
                                         f(bs['leak_f1'].loc[k, 'mean_inf'], sign=True), f(bs['leak_pr'].loc[k, 'mean_inf'], sign=True)]) + ' \\\\')
T['leak'] = table('Controlled leakage injection under random 5-fold CV (3 folds, Set~C). Each row re-introduces one construction error '
                  'that SAFE-G removes; entries are the inflation relative to the leakage-free pipeline, averaged over LR, LightGBM and RF '
                  '(LightGBM shown separately for IEEE-CIS). Leakage-free baselines: IEEE-CIS F1 ' + f(ie['leak_f1'].loc['sizintisiz', 'mean'])
                  + ', PR-AUC ' + f(ie['leak_pr'].loc['sizintisiz', 'mean']) + '; BankSim F1 ' + f(bs['leak_f1'].loc['sizintisiz', 'mean'])
                  + ', PR-AUC ' + f(bs['leak_pr'].loc['sizintisiz', 'mean']) + '.',
                  'tab:leak', 'lccccc',
                  ['& \\multicolumn{3}{c}{IEEE-CIS} & \\multicolumn{2}{c}{BankSim} \\\\',
                   '\\cmidrule{2-4}\\cmidrule{5-6}',
                   'Injected leakage path & $\\Delta$F1 & $\\Delta$PR-AUC & $\\Delta$F1 (LightGBM) & $\\Delta$F1 & $\\Delta$PR-AUC \\\\'], rows)

# ---------------------------------------------------------------- protocol dependence
PN = {'P1': 'P1 random CV', 'P2': 'P2 entity-grouped', 'P3': 'P3 temporal', 'P4': 'P4 rolling-origin'}
def holm(src, p, met):
    b = src['b9'].get((p, 'D (A+graf) vs A (ham)', met)) if 'b9' in src else None
    return f"{b['holm']}/{b['n']}" if b else '--'
def holm_e(met):
    b = el['b9_e2'][('D (A+graf) vs A (ham)', met)]; return f"{b['holm']}/{b['n']}"
rows = []
for ds, name, src in [('ieee', 'IEEE-CIS', ie), ('ell', 'Elliptic++', el), ('bs', 'BankSim', bs)]:
    first = True
    for p in ['P1', 'P2', 'P3', 'P4']:
        if p not in S[ds]: continue
        s = S[ds][p]
        if ds == 'ell':
            hf, hp = (holm_e('f1'), holm_e('pr_auc')) if p == 'P1' else ('--', '--')
        else:
            hf, hp = (holm(src, p, 'f1'), holm(src, p, 'pr_auc')) if p in ('P1', 'P2') else ('--', '--')
        rows.append(f"{name if first else ''} & {PN[p]} & {f(s['kappa'], 2)} & {s['nm']} & {f(s['f1'], sign=True)} & {hf} & {f(s['pr'], sign=True)} & {hp} \\\\")
        first = False
    rows.append('\\addlinespace')
rows = rows[:-1]
T['protocol'] = table('The contribution of graph features (Set D minus Set A, averaged over models) under each validation protocol. '
                      '$\\bar\\kappa$: mean entity coverage of the evaluation partition. Holm: models for which the difference is significant '
                      'under the corrected resampled $t$-test with Holm adjustment ($\\alpha=0.05$); not computed for single-split and '
                      'rolling-origin protocols, whose folds are not exchangeable. P4 on IEEE-CIS uses LightGBM and XGBoost only.',
                      'tab:protocol', 'llcccccc',
                      ['Dataset & Protocol & $\\bar\\kappa$ & Models & $\\Delta$F1 & Holm & $\\Delta$PR-AUC & Holm \\\\'], rows, wide=True)

# ---------------------------------------------------------------- naive vs corrected
rows = []
CMP = [('B (A+davranışsal) vs A (ham)', 'B$-$A'), ('C (B+graf) vs B (A+davranışsal)', 'C$-$B'), ('D (A+graf) vs A (ham)', 'D$-$A')]
for name, src in [('IEEE-CIS', ie), ('BankSim', bs)]:
    first = True
    for p in ['P1', 'P2']:
        for c, cs in CMP:
            a, b = src['b9'][(p, c, 'f1')], src['b9'][(p, c, 'pr_auc')]
            rows.append(f"{name if first else ''} & {p} & {cs} & {f(a['mean'], sign=True)} & {a['naive']} & {a['holm']} & "
                        f"{f(b['mean'], sign=True)} & {b['naive']} & {b['holm']} \\\\")
            first = False
    rows.append('\\addlinespace')
first = True
for c, cs in CMP:
    a, b = el['b9_e2'][(c, 'f1')], el['b9_e2'][(c, 'pr_auc')]
    rows.append(f"{'Elliptic++' if first else ''} & P1 & {cs} & {f(a['mean'], sign=True)} & {a['naive']} & {a['holm']} & "
                f"{f(b['mean'], sign=True)} & {b['naive']} & {b['holm']} \\\\")
    first = False
T['sig'] = table('Number of models with a significant difference under the naive paired $t$-test and under the corrected resampled $t$-test '
                 'with Holm adjustment ($\\alpha=0.05$; 10 models for IEEE-CIS and BankSim, 4 for Elliptic++).',
                 'tab:sig', 'lllcccccc',
                 ['& & & \\multicolumn{3}{c}{F1} & \\multicolumn{3}{c}{PR-AUC} \\\\', '\\cmidrule{4-6}\\cmidrule{7-9}',
                  'Dataset & Prot. & Comparison & $\\bar\\Delta$ & naive & Holm & $\\bar\\Delta$ & naive & Holm \\\\'], rows, wide=True)

# ---------------------------------------------------------------- kappa-stratified
def cnt(c, tag, sub): d = c[(tag, sub)]; return d
rows = []
cP3s, cP3u = ie['counts'][('C_P3', 'gorulen')], ie['counts'][('C_P3', 'gorulmeyen')]
cP4s, cP4u = ie['p4_counts'][('R_P4', 'gorulen')], ie['p4_counts'][('R_P4', 'gorulmeyen')]
k = ie['kpr']
rows.append(f"IEEE-CIS P3 (5 models) & seen & {cP3s['n']:,} / {cP3s['pos']} & {f(k.loc[('C_P3','gorulen','A (ham)'),'mean'])} & "
            f"{f(k.loc[('C_P3','gorulen','C (B+graf)'),'mean'])} & {f(k.loc[('C_P3','gorulen','C (B+graf) [κ-mask]'),'mean'])} & -- & -- \\\\")
rows.append(f" & unseen & {cP3u['n']:,} / {cP3u['pos']} & {f(k.loc[('C_P3','gorulmeyen','A (ham)'),'mean'])} & "
            f"{f(k.loc[('C_P3','gorulmeyen','C (B+graf)'),'mean'])} & {f(k.loc[('C_P3','gorulmeyen','C (B+graf) [κ-mask]'),'mean'])} & -- & -- \\\\")
p4 = ie['p4_k']
for sub, cc in [('gorulen', cP4s), ('gorulmeyen', cP4u)]:
    rows.append(f"{'IEEE-CIS P4 (2 models)' if sub == 'gorulen' else ''} & {'seen' if sub == 'gorulen' else 'unseen'} & {cc['n']:,} / {cc['pos']} & "
                f"{f(p4.loc[(sub,'A (ham)'),'mean'])} & {f(p4.loc[(sub,'C (B+graf)'),'mean'])} & {f(p4.loc[(sub,'C (B+graf) [κ-mask]'),'mean'])} & "
                f"{f(p4.loc[(sub,'A_full (A8)'),'mean'])} & {f(p4.loc[(sub,'C_full (A8)'),'mean'])} \\\\")
ek = el['kpr']; e4c = el['e4_counts']
for sub in ['gorulen', 'gorulmeyen']:
    cc = e4c[('E4_P4', sub)]
    rows.append(f"{'Elliptic++ P4 (4 models)' if sub == 'gorulen' else ''} & {'seen' if sub == 'gorulen' else 'unseen'} & {cc['n']:,} / {cc['pos']} & "
                f"{f(ek.loc[('E2_P4',sub,'A (ham)'),'mean'])} & {f(ek.loc[('E2_P4',sub,'C (B+graf)'),'mean'])} & "
                f"{f(ek.loc[('E2_P4',sub,'C (B+graf) [κ-mask]'),'mean'])} & {f(ek.loc[('E2_P4',sub,'A+AGG (referans)'),'mean'])} & {f(el['k4pr'].loc[('E4_P4',sub,'AA+C'),'mean'])} \\\\")
T['kappa'] = table('Entity-coverage-stratified PR-AUC: evaluation transactions whose entity is (seen) or is not (unseen) in the training graph. '
                   '$n$/pos: pooled number of evaluation transactions and positives over folds. Strong baseline: A$_\\text{full}$ (all raw IEEE-CIS '
                   'fields) or A+AGG (Elliptic++ local and provided neighbourhood-aggregate features). In Elliptic++ P4 one origin contributes only '
                   'two seen positives; in P3 (not shown) the seen subset holds five positives and is not interpretable.',
                   'tab:kappa', 'llrccccc',
                   ['& & & \\multicolumn{3}{c}{Narrow baseline} & \\multicolumn{2}{c}{Strong baseline} \\\\', '\\cmidrule{4-6}\\cmidrule{7-8}',
                    'Dataset, protocol & Subset & $n$ / pos & A & C & C $\\kappa$-mask & base & +graph (C) \\\\'], rows, wide=True)

# ---------------------------------------------------------------- remedies (narrow baseline)
rows = []
cf, cp = ie['c_f1'], ie['c_pr']
for lab_, p in [('Random 5-fold CV', 'P1'), ('GroupKFold (uid)', 'P2'), ('Temporal out-of-time', 'P3')]:
    r1, r2 = cf.loc[(lab_, 'ORTALAMA')], cp.loc[(lab_, 'ORTALAMA')]
    rows.append(f"{'IEEE-CIS' if p == 'P1' else ''} & {p} & {f(r1['kappa'], 2)} & " + ' & '.join(f(r1[c]) for c in ['A', 'C', 'Ck', 'RC']) + ' & '
                + ' & '.join(f(r2[c]) for c in ['A', 'C', 'Ck', 'RC']) + ' \\\\')
m4f = ie['p4_f1'].mean(axis=1); m4p = ie['p4_pr'].mean(axis=1)
rows.append(f" & P4$^{{\\ddagger}}$ & {f(S['ieee']['P4']['kappa'], 2)} & " + ' & '.join(f(m4f[c]) for c in ['A (ham)', 'C (B+graf)', 'C (B+graf) [κ-mask]', 'Router C (B+graf) | A (ham)'])
            + ' & ' + ' & '.join(f(m4p[c]) for c in ['A (ham)', 'C (B+graf)', 'C (B+graf) [κ-mask]', 'Router C (B+graf) | A (ham)']) + ' \\\\')
rows.append('\\addlinespace')
ef, ep = el['e2_f1'], el['e2_pr']
for lab_, p in [('Random 5-fold CV', 'P1'), ('Temporal 1-34 / 35-49', 'P3'), ('Rolling-origin', 'P4')]:
    r1, r2 = ef.loc[(lab_, 'ORTALAMA')], ep.loc[(lab_, 'ORTALAMA')]
    rows.append(f"{'Elliptic++' if p == 'P1' else ''} & {p} & {f(r1['kappa'], 2)} & " + ' & '.join(f(r1[c]) for c in ['A', 'C', 'Ck', 'RC']) + ' & '
                + ' & '.join(f(r2[c]) for c in ['A', 'C', 'Ck', 'RC']) + ' \\\\')
T['remedy'] = table('Coverage-aware remedies on the narrow baseline (means over LR, RF, XGBoost, LightGBM and MLP for IEEE-CIS; LR, RF, XGBoost and '
                    'LightGBM for Elliptic++). Router: graph-conditioned model (C) if the entity is in the training graph, graph-free model (A) '
                    'otherwise. Under P2, $\\kappa=0$ and the router reduces to A by construction. $^{\\ddagger}$\\,LightGBM and XGBoost only.',
                    'tab:remedy', 'llccccccccc',
                    ['& & & \\multicolumn{4}{c}{F1} & \\multicolumn{4}{c}{PR-AUC} \\\\', '\\cmidrule{4-7}\\cmidrule{8-11}',
                     'Dataset & Prot. & $\\bar\\kappa$ & A & C & C$\\kappa$ & Router & A & C & C$\\kappa$ & Router \\\\'], rows, wide=True)

# ---------------------------------------------------------------- headline: rolling-origin strong baseline
rows = []
def cell(d): return f"{f(d['mean'], sign=True)} $\\pm$ {d['sd']:.3f} ({d['wins']})"
for m in ['LightGBM', 'XGBoost']:
    rows.append(f"{'IEEE-CIS' if m == 'LightGBM' else ''} & {m} & {cell(ie['p4_dpr'][('D_full−A_full', m)])} & "
                f"{cell(ie['p4_dpr'][('C_fullκ − C_full', m)])} & {cell(ie['p4_dpr'][('Router C_full|A_full − A_full', m)])} \\\\")
rows.append('\\addlinespace')
for m in ['LR', 'RF', 'XGBoost', 'LightGBM']:
    rows.append(f"{'Elliptic++' if m == 'LR' else ''} & {m} & {cell(el['e4_wins_pr'][('D−AA', m)])} & "
                f"{cell(el['e4_wins_pr'][('Cκ−C', m)])} & {cell(el['e4_wins_pr'][('RouterC−AA', m)])} \\\\")
T['p4strong'] = table('Rolling-origin evaluation (P4, five expanding-window origins) on the strong baseline: PR-AUC differences, mean $\\pm$ s.d. over '
                      'origins, with the number of origins in which the difference is positive. Base: A$_\\text{full}$ (IEEE-CIS) or A+AGG '
                      '(Elliptic++). Concatenation: base + graph (D); $\\kappa$-mask: coverage-aware training of base + behavioural + graph (C) '
                      'relative to its standard version; Router: base + behavioural + graph (C) for seen entities, base otherwise.',
                      'tab:p4strong', 'llccc',
                      ['Dataset & Model & Concatenation $-$ base & $\\kappa$-mask $-$ standard & Router $-$ base \\\\'], rows, wide=True)

# ---------------------------------------------------------------- GNN
rows = []
order = [('Temporal 1-34 / 35-49', 'P3'), ('Rolling-origin', 'P4'), ('Random 5-fold CV', 'P1')]
g = el['gnn']
for fs in ['GCN (A)', 'GCN (A+AGG)', 'GraphSAGE (A)', 'GraphSAGE (A+AGG)']:
    rows.append(f'{fs} & ' + ' & '.join(f(g.loc[(pl, fs), 'pr']) for pl, _ in order) + ' & '
                + ' & '.join(f(g.loc[(pl, fs), 'f1']) for pl, _ in order) + ' \\\\')
rows.append('\\midrule')
for mdl, setname, key in [('LightGBM', 'A', 'A'), ('LightGBM', 'A+AGG', 'AA'), ('XGBoost', 'A+AGG', 'AA'), ('LightGBM', 'Router C$\\mid$A', 'RC'),
                          ('XGBoost', 'Router C$\\mid$A', 'RC')]:
    rows.append(f'{mdl} {setname} & ' + ' & '.join(f(el['e2_pr'].loc[(pl, mdl), key]) for pl, _ in order) + ' & '
                + ' & '.join(f(el['e2_f1'].loc[(pl, mdl), key]) for pl, _ in order) + ' \\\\')
T['gnn'] = table('GNNs and tabular models on Elliptic++ under the same partitions and threshold rule. GNNs: two-layer GCN and GraphSAGE on the '
                 'transaction graph, trained inductively on the structure partition, means over three seeds (seed s.d. $\\le 0.03$).',
                 'tab:gnn', 'lcccccc',
                 ['& \\multicolumn{3}{c}{PR-AUC} & \\multicolumn{3}{c}{F1} \\\\', '\\cmidrule{2-4}\\cmidrule{5-7}',
                  'Method & P3 & P4 & P1 & P3 & P4 & P1 \\\\'], rows)

# ---------------------------------------------------------------- resampling
rows = []
for k in ['None', 'RandomUnder', 'NearMiss', 'SMOTE', 'SMOTETomek']:
    a, b = ie['samp'].loc[k], bs['samp'].loc[k]
    nm = {'None': 'None', 'RandomUnder': 'Random undersampling', 'NearMiss': 'NearMiss-1', 'SMOTE': 'SMOTE', 'SMOTETomek': 'SMOTE + Tomek'}[k]
    rows.append(f'{nm} & ' + ' & '.join(f(a[c]) for c in ['prec', 'rec', 'f1', 'pr', 'fpr']) + ' & ' + ' & '.join(f(b[c]) for c in ['prec', 'rec', 'f1', 'pr', 'fpr']) + ' \\\\')
T['samp'] = table('Resampling strategies under in-fold threshold selection (RF, Set C, random CV; all strategies use the same 1:3 target ratio).',
                  'tab:samp', 'lcccccccccc',
                  ['& \\multicolumn{5}{c}{IEEE-CIS} & \\multicolumn{5}{c}{BankSim} \\\\', '\\cmidrule{2-6}\\cmidrule{7-11}',
                   'Strategy & Prec. & Rec. & F1 & PR-AUC & FPR & Prec. & Rec. & F1 & PR-AUC & FPR \\\\'], rows, wide=True)

# ---------------------------------------------------------------- appendix: per-model
def per_model(src, keys, cap, lab):
    rows = []
    for m in N.MODELS10:
        cells = []
        for k in keys:
            t = src[k]
            cells += [f(t.loc[m, 'A']), f(t.loc[m, 'B']), f(t.loc[m, 'C']), f(t.loc[m, 'D'])]
        rows.append(f'{m} & ' + ' & '.join(cells) + ' \\\\')
    hdr = ['& ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{h}}}' for h in ['P1 random CV', 'P2 entity-grouped', 'P3 temporal']) + ' \\\\',
           '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}',
           'Model & ' + ' & '.join(['A & B & C & D'] * 3) + ' \\\\']
    return table(cap, lab, 'l' + 'c' * 12, hdr, rows, wide=True, size='\\footnotesize', pos='!htbp')
T['app_ieee'] = per_model(ie, ['p1_f1', 'p2_f1', 'p3_f1'], 'IEEE-CIS: F1 per model, feature set and protocol.', 'tab:app_ieee')
T['app_bs'] = per_model(bs, ['p1_f1', 'p2_f1', 'p3_f1'], 'BankSim: F1 per model, feature set and protocol.', 'tab:app_bs')

def app_ell():
    rows = []
    for lab_, p in [('Random 5-fold CV', 'P1'), ('Temporal 1-34 / 35-49', 'P3'), ('Rolling-origin', 'P4')]:
        for m in ['LR', 'RF', 'XGBoost', 'LightGBM']:
            r = el['e2_pr'].loc[(lab_, m)]; s = el['e4_pr'].loc[(lab_, m)]
            rows.append(f"{p if m == 'LR' else ''} & {m} & " + ' & '.join(f(r[c]) for c in ['A', 'B', 'C', 'D', 'Ck', 'RC']) + ' & '
                        + ' & '.join(f(s[c]) for c in ['AA', 'AA+D', 'AA+Ck', 'RAC']) + ' \\\\')
        rows.append('\\addlinespace')
    return table('Elliptic++: PR-AUC per model, feature set and protocol. Left: narrow baseline; right: strong baseline (AA = A+AGG). Routers combine C (or AA+C) for seen entities with A (or AA).',
                 'tab:app_ell', 'llcccccccccc',
                 ['& & \\multicolumn{6}{c}{Narrow baseline} & \\multicolumn{4}{c}{Strong baseline} \\\\', '\\cmidrule{3-8}\\cmidrule{9-12}',
                  'Prot. & Model & A & B & C & D & C$\\kappa$ & Router & AA & AA+D & AA+C$\\kappa$ & Router \\\\'], rows[:-1], wide=True, size='\\scriptsize', pos='!htbp')
T['app_ell'] = app_ell()

def app_origin():
    rows = []
    for r in ie['p4_origin_pr']:
        rows.append(f"IEEE-CIS & {r[1]} & {f(float(r[2]), 2)} & {r[3]} & " + ' & '.join(f(float(x), sign=True) for x in [r[4], r[5], r[6], r[8]]) + ' \\\\')
    rows.append('\\addlinespace')
    e2o, e4o = el['e2_origin_pr'], el['e4_origin_pr']
    for idx in e2o.index:
        rows.append(f"Elliptic++ & {idx.split()[1]} & {f(e2o.loc[idx,'kappa'], 2)} & {int(e2o.loc[idx,'npos'])} & {f(e2o.loc[idx,'D-A'], sign=True)} & "
                    f"{f(e2o.loc[idx,'RC-A'], sign=True)} & {f(e4o.loc[idx,'D-AA'], sign=True)} & {f(e4o.loc[idx,'RC-AA'], sign=True)} \\\\")
    return table('Rolling-origin PR-AUC differences per origin (means over models). IEEE-CIS origins are training-window quantiles of '
                 'transaction time; Elliptic++ origins are the last training time step. Base: A$_\\text{full}$ or A+AGG.',
                 'tab:app_origin', 'llccrrrr',
                 ['& & & & \\multicolumn{2}{c}{Narrow baseline} & \\multicolumn{2}{c}{Strong baseline} \\\\', '\\cmidrule{5-6}\\cmidrule{7-8}',
                  'Dataset & Origin & $\\kappa$ & pos & D$-$A & Router$-$A & D$-$base & Router$-$base \\\\'], rows, wide=True, size='\\footnotesize', pos='!htbp')
T['app_origin'] = app_origin()

if __name__ == '__main__':
    for k, v in T.items():
        print(f'%%%%% {k}\n{v}\n')
