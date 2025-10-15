import pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

def load_csv(path):
    return pd.read_csv(path)

def plot_learning_curve(csv_paths, labels, out_path):
    plt.figure(figsize=(8,4))
    for p,label in zip(csv_paths, labels):
        df = load_csv(p)
        dfm = df.groupby('episode')['return'].mean().rolling(5).mean().reset_index()
        plt.plot(dfm['episode'], dfm['return'], label=label)
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def violin_plot(csv_paths, labels, out_path):
    combined = []
    for p,label in zip(csv_paths, labels):
        df = load_csv(p)
        seeds = df.groupby('seed')['return'].sum().reset_index()
        seeds['alg'] = label
        combined.append(seeds[['alg','return']])
    df_all = pd.concat(combined, axis=0)
    plt.figure(figsize=(6,4)); sns.violinplot(x='alg', y='return', data=df_all); plt.tight_layout(); plt.savefig(out_path); plt.close()

def compute_stats(csv_paths, labels):
    stats = []
    for p,label in zip(csv_paths, labels):
        df = load_csv(p)
        vals = df.groupby('seed')['return'].sum().values
        m = vals.mean(); se = stats_sem(vals); ci = se * 1.96 if len(vals)>1 else 0.0
        stats.append({'label': label, 'mean': m, 'ci95': ci})
    return stats

def stats_sem(x):
    import numpy as np
    from scipy import stats
    return stats.sem(x) if len(x)>1 else 0.0

def generate_all_figures_and_pdf(cfg):
    os.makedirs('results/figures', exist_ok=True)
    csvs = [cfg['ours_out'] + '/metrics.csv'] + [b['out_csv'] for b in cfg.get('baselines',[])]
    labels = ['Ours'] + [b['name'] for b in cfg.get('baselines',[])]
    # ensure csvs exist
    for c in csvs:
        if not os.path.exists(c):
            # skip if missing
            print('Missing', c)
    # plot learning curve
    plot_learning_curve(csvs, labels, 'results/figures/learning_curve.png')
    violin_plot(csvs, labels, 'results/figures/violin.png')
    # generate PDF summary
    doc = SimpleDocTemplate('results/results_summary.pdf', pagesize=letter)
    styles = getSampleStyleSheet()
    elems = [Paragraph('Results Summary', styles['Title']), Spacer(1,12)]
    elems.append(Paragraph('Figures', styles['Heading2']))
    elems.append(Image('results/figures/learning_curve.png', width=400, height=200))
    elems.append(Spacer(1,12))
    elems.append(Image('results/figures/violin.png', width=400, height=200))
    elems.append(Spacer(1,12))
    # stats table
    stats = compute_stats(csvs, labels)
    data = [['Alg', 'Mean Return', '95% CI']] + [[s['label'], f"{s['mean']:.2f}", f"{s['ci95']:.2f}"] for s in stats]
    table = Table(data)
    elems.append(table)
    doc.build(elems)
    print('Generated results/results_summary.pdf')
