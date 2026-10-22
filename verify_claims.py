"""Metindeki temel iddialari dondurulmus sayilardan yeniden hesaplayip metindeki degerle karsilastirir."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, safeg_numbers as N
D = N.D; ie, el, bs, S = D['ieee'], D['ell'], D['bs'], D['summary']
ok = True
def chk(name, value, text_value, tol=0.00051):
    global ok
    good = abs(value - text_value) <= tol
    ok &= good
    print(f"{'OK ' if good else 'XX '} {name:60s} hesap={value:+.4f} metin={text_value:+.4f}")
kp, k4 = ie['kpr'], el['kpr']
chk('Elliptic P1 D-A F1', S['ell']['P1']['f1'], 0.053); chk('Elliptic P3 D-A F1', S['ell']['P3']['f1'], -0.082)
chk('Elliptic P4 D-A F1', S['ell']['P4']['f1'], -0.048); chk('IEEE P1 D-A F1', S['ieee']['P1']['f1'], 0.029)
chk('IEEE P3 PR D-A', S['ieee']['P3']['pr'], 0.014); chk('IEEE P4 PR D-A', S['ieee']['P4']['pr'], 0.042)
chk('Ell P3 PR D-A', S['ell']['P3']['pr'], 0.012); chk('Ell P4 PR D-A', S['ell']['P4']['pr'], 0.014)
chk('IEEE leak total F1', ie['leak_f1'].loc['hepsi', 'mean_inf'], 0.042); chk('IEEE leak total PR', ie['leak_pr'].loc['hepsi', 'mean_inf'], 0.037)
chk('IEEE leak LGBM', ie['leak_f1'].loc['hepsi', 'LightGBM_inf'], 0.089); chk('BS leak total F1', bs['leak_f1'].loc['hepsi', 'mean_inf'], 0.012)
chk('IEEE P3 seen A', kp.loc[('C_P3', 'gorulen', 'A (ham)'), 'mean'], 0.172); chk('IEEE P3 seen C', kp.loc[('C_P3', 'gorulen', 'C (B+graf)'), 'mean'], 0.350)
chk('IEEE P3 seen Ck', kp.loc[('C_P3', 'gorulen', 'C (B+graf) [κ-mask]'), 'mean'], 0.185)
p4 = ie['p4_k']
chk('IEEE P4 seen A', p4.loc[('gorulen', 'A (ham)'), 'mean'], 0.305); chk('IEEE P4 seen C', p4.loc[('gorulen', 'C (B+graf)'), 'mean'], 0.578)
chk('IEEE P4 seen Afull', p4.loc[('gorulen', 'A_full (A8)'), 'mean'], 0.454); chk('IEEE P4 seen Cfull', p4.loc[('gorulen', 'C_full (A8)'), 'mean'], 0.605)
chk('Ell P4 seen A', k4.loc[('E2_P4', 'gorulen', 'A (ham)'), 'mean'], 0.400); chk('Ell P4 seen C', k4.loc[('E2_P4', 'gorulen', 'C (B+graf)'), 'mean'], 0.733)
chk('Ell P4 seen AA', k4.loc[('E2_P4', 'gorulen', 'A+AGG (referans)'), 'mean'], 0.469); chk('Ell P4 seen AA+C', el['k4pr'].loc[('E4_P4', 'gorulen', 'AA+C'), 'mean'], 0.751)
drops = [kp.loc[('C_P3', 'gorulmeyen', 'A (ham)'), 'mean'] - kp.loc[('C_P3', 'gorulmeyen', 'C (B+graf)'), 'mean'],
         p4.loc[('gorulmeyen', 'A (ham)'), 'mean'] - p4.loc[('gorulmeyen', 'C (B+graf)'), 'mean'],
         p4.loc[('gorulmeyen', 'A_full (A8)'), 'mean'] - p4.loc[('gorulmeyen', 'C_full (A8)'), 'mean'],
         k4.loc[('E2_P4', 'gorulmeyen', 'A (ham)'), 'mean'] - k4.loc[('E2_P4', 'gorulmeyen', 'C (B+graf)'), 'mean'],
         k4.loc[('E2_P4', 'gorulmeyen', 'A+AGG (referans)'), 'mean'] - el['k4pr'].loc[('E4_P4', 'gorulmeyen', 'AA+C'), 'mean']]
chk('unseen drop min', min(drops), 0.018); chk('unseen drop max', max(drops), 0.047)
c = ie['counts']; chk('IEEE P3 seen prev %', c[('C_P3', 'gorulen')]['prev'], 1.5, 0.05); chk('IEEE P3 unseen prev %', c[('C_P3', 'gorulmeyen')]['prev'], 4.5, 0.05)
e = el['e4_counts']; chk('Ell P4 seen prev', e[('E4_P4', 'gorulen')]['prev'], 4.4, 0.05); chk('Ell P4 unseen prev', e[('E4_P4', 'gorulmeyen')]['prev'], 13.3, 0.05)
chk('Ell P3 seen pos', e[('E4_P3', 'gorulen')]['pos'], 5); chk('Ell P4 min seen pos', e[('E4_P4', 'gorulen')]['minpos'], 2)
cf, cp = ie['c_f1'], ie['c_pr']
for col, v in [('A', 0.275), ('C', 0.246), ('RC', 0.296)]: chk(f'IEEE P3 F1 {col}', cf.loc[('Temporal out-of-time', 'ORTALAMA'), col], v)
for col, v in [('A', 0.202), ('C', 0.221), ('RC', 0.240)]: chk(f'IEEE P3 PR {col}', cp.loc[('Temporal out-of-time', 'ORTALAMA'), col], v)
ef, ep = el['e2_f1'], el['e2_pr']
for col, v in [('A', 0.670), ('C', 0.581), ('RC', 0.701)]: chk(f'Ell P4 F1 {col}', ef.loc[('Rolling-origin', 'ORTALAMA'), col], v)
for col, v in [('A', 0.690), ('C', 0.692), ('RC', 0.732)]: chk(f'Ell P4 PR {col}', ep.loc[('Rolling-origin', 'ORTALAMA'), col], v)
chk('Ell P3 F1 C', ef.loc[('Temporal 1-34 / 35-49', 'ORTALAMA'), 'C'], 0.532); chk('Ell P3 F1 Ck', ef.loc[('Temporal 1-34 / 35-49', 'ORTALAMA'), 'Ck'], 0.667)
w = ie['p4_dpr']
chk('IEEE P4 concat LGBM', w[('D_full−A_full', 'LightGBM')]['mean'], 0.017); chk('IEEE P4 concat XGB', w[('D_full−A_full', 'XGBoost')]['mean'], 0.015)
chk('IEEE P4 router LGBM', w[('Router C_full|A_full − A_full', 'LightGBM')]['mean'], 0.040); chk('IEEE P4 router XGB', w[('Router C_full|A_full − A_full', 'XGBoost')]['mean'], 0.029)
ew = el['e4_wins_pr']
chk('Ell router LR', ew[('RouterC−AA', 'LR')]['mean'], 0.120)
tree = [ew[('RouterC−AA', m)]['mean'] for m in ['RF', 'XGBoost', 'LightGBM']]; chk('Ell router tree min', min(tree), 0.010); chk('Ell router tree max', max(tree), 0.016)
print('router wins all:', all(v['wins'] == '5/5' for (k, m), v in list(ew.items()) + list(w.items()) if k.startswith('Router') and 'D' not in k[:8] or k in ('RouterC−AA', 'Router C_full|A_full − A_full')))
o = [float(r[6]) for r in ie['p4_origin_pr']]; chk('IEEE origin Dfull min', min(o), 0.004); chk('IEEE origin Dfull max', max(o), 0.029)
chk('IEEE A8 P3 D-A PR', ie['a8_pr'].loc['Temporal out-of-time LightGBM', 'D-A'] if 'Temporal out-of-time LightGBM' in ie['a8_pr'].index else ie['a8_pr'].iloc[2]['D-A'], 0.074)
g = el['gnn']
chk('GNN best P4', g.xs('Rolling-origin')['pr'].max(), 0.719); chk('GNN best P3', g.xs('Temporal 1-34 / 35-49')['pr'].max(), 0.719)
chk('GNN best P1', g.xs('Random 5-fold CV')['pr'].max(), 0.944); chk('XGB AA P4', ep.loc[('Rolling-origin', 'XGBoost'), 'AA'], 0.851)
chk('LGBM AA P3', ep.loc[('Temporal 1-34 / 35-49', 'LightGBM'), 'AA'], 0.798)
chk('best tabular P1 PR', max(ep.xs('Random 5-fold CV')[['A', 'B', 'C', 'D', 'AA', 'Ck', 'Dk', 'RC', 'RD']].drop('ORTALAMA').max()), 0.992)
chk('thr 0.5 F1', ie['thr']['f1'][0], 0.092); chk('thr sel F1', ie['thr']['f1'][1], 0.326); chk('thr', ie['thr_value'], 0.19, 0.001)
at = ie['alg_time']; chk('betweenness share of extraction time %', 100 * at['betweenness'] / ie['total_extract'], 93, 0.6)
chk('BS shap graph', bs['shap']['Graph'], 64.0, 0.05); chk('IEEE shap graph', ie['shap']['Graph'], 12.1, 0.05)
print('\nGENEL:', 'TUM IDDIALAR TUTARLI' if ok else 'UYUSMAZLIK VAR')
