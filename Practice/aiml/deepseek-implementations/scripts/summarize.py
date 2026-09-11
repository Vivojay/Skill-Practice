"""Regenerate the small paired-investigation table from saved JSON summaries."""
import csv
import json
from pathlib import Path
import statistics


def main():
    rows=[]
    for seed in (11,22,33):
        pair={v:json.loads(Path(f'results/moe-{v}-{seed}.json').read_text()) for v in ('instant','ema')}
        for variant,r in pair.items():
            rows.append(dict(seed=seed,variant=variant,mse=r['heldout_mse'],imbalance=r['late_imbalance'],
                             switching=r['switch_rate'],adaptation=r['adaptation_updates'],
                             step_ms=r['step_time']['median_ms'],controller_ms=r['controller_time']['median_ms']))
    with Path('results/routing-pairs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    averages={v:{key:statistics.mean(r[key] for r in rows if r['variant']==v)
                 for key in ['mse','imbalance','switching','step_ms','controller_ms']} for v in ['instant','ema']}
    result=dict(paired_seeds=[11,22,33],means=averages,
                relative_mse_change=averages['ema']['mse']/averages['instant']['mse']-1,
                relative_imbalance_change=averages['ema']['imbalance']/averages['instant']['imbalance']-1,
                conclusion='Inconclusive overall: balance/switching and quality criteria pass in the mean; seed 11 adaptation is right censored in both arms.',
                novelty='Known smoothing ablation, no novelty claim',
                uncertainty='Three paired seeds, fixed arm order; no significance or robust speedup claim')
    Path('results/routing-summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
