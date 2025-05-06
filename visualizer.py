#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats

def load_and_flatten(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data.get('results', []))
    df.columns = [c.replace('.', '_') for c in df.columns]
    return df

def prepare_data(df):
    # Detect columns dynamically
    time_cols = [c for c in df.columns if c.endswith('_time_s')]
    mem_cols  = [c for c in df.columns if c.endswith('_rss_kB')]
    correct_col = 'equivalent'

    # Melt for plotting
    id_vars = [col for col in ['pass','n'] if col in df.columns]
    time_df = df.melt(id_vars=id_vars, value_vars=time_cols,
                      var_name='algorithm', value_name='time_s')
    mem_df  = df.melt(id_vars=id_vars, value_vars=mem_cols,
                      var_name='algorithm', value_name='rss_kB')
    for d in (time_df, mem_df):
        d['algorithm'] = d['algorithm'].str.replace(r'_(time_s|rss_kB)$','',regex=True)

    # Drop pass 0 entirely
    time_df = time_df[time_df['pass'] != 0]
    mem_df  = mem_df[mem_df['pass'] != 0]

    # correctness tracking (ignore pass 0)
    if correct_col in df.columns:
        corr = df[id_vars + [correct_col]].copy().rename(columns={correct_col:'correct'})
        if 'pass' in corr.columns:
            corr = corr[corr['pass'] != 0]
    else:
        corr = pd.DataFrame(columns=id_vars + ['correct'])

    # drop NaNs
    time_df = time_df.dropna(subset=['time_s'])
    mem_df  = mem_df.dropna(subset=['rss_kB'])

    # aggregates
    stats_time_pass = time_df.groupby(['pass','algorithm'])['time_s'].mean().reset_index()
    stats_mem_pass  = mem_df.groupby(['pass','algorithm'])['rss_kB'].mean().reset_index()

    # per-n & algorithm: mean and std across passes
    stats_time_n = time_df.groupby(['n','algorithm'])['time_s'].agg(['mean','std']).reset_index()
    stats_mem_n  = mem_df.groupby(['n','algorithm'])['rss_kB'].agg(['mean','std']).reset_index()

    # table of raw results (exclude pass 0 rows)
    table = df.copy()
    if 'pass' in table.columns:
        table = table[table['pass'] != 0]

    return time_df, mem_df, stats_time_pass, stats_mem_pass, stats_time_n, stats_mem_n, corr, table

def build_gui(time_df, mem_df, stp, smp, stn, smn, corr, table, dpi):
    root = tk.Tk()
    root.title("Benchmark Visualizer for Matrix Multiplication Algorithms")
    root.attributes('-zoomed', True) if hasattr(root, 'attributes') else root.state('normal')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    passes = sorted(time_df['pass'].unique()) if 'pass' in time_df else []

    def redraw(fig, df, x, y, title, xlabel, ylabel):
        fig.clf()
        ax = fig.add_subplot(111)
        for alg, grp in df.groupby('algorithm'):
            ax.plot(grp[x], grp[y], marker='o', label=alg)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.grid(True, which='both', ls='--', lw=0.5)
        ax.legend(title='Algorithm')

    # Time Tab
    tab1 = ttk.Frame(notebook); notebook.add(tab1, text='Time (s)')
    if passes:
        combo_time = ttk.Combobox(tab1, values=passes, state='readonly'); combo_time.set(passes[0]); combo_time.pack(anchor='nw', pady=5, padx=5)
    fig1 = Figure(figsize=(6,4), dpi=dpi); canvas1 = FigureCanvasTkAgg(fig1, master=tab1); canvas1.get_tk_widget().pack(fill='both', expand=True); NavigationToolbar2Tk(canvas1, tab1)
    redraw(fig1, time_df[time_df['pass']==passes[0]] if passes else time_df, 'n','time_s','Time vs n','n','Time (s)'); canvas1.draw()
    if passes: combo_time.bind('<<ComboboxSelected>>', lambda e: (redraw(fig1, time_df[time_df['pass']==int(combo_time.get())],'n','time_s',f'Time vs n (pass {combo_time.get()})','n','Time (s)'), canvas1.draw()))

    # Memory Tab
    tab2 = ttk.Frame(notebook); notebook.add(tab2, text='Memory (kB)')
    if passes:
        combo_mem = ttk.Combobox(tab2, values=passes, state='readonly'); combo_mem.set(passes[0]); combo_mem.pack(anchor='nw', pady=5, padx=5)
    fig2 = Figure(figsize=(6,4), dpi=dpi); canvas2 = FigureCanvasTkAgg(fig2, master=tab2); canvas2.get_tk_widget().pack(fill='both', expand=True); NavigationToolbar2Tk(canvas2, tab2)
    redraw(fig2, mem_df[mem_df['pass']==passes[0]] if passes else mem_df,'n','rss_kB','Memory vs n','n','Memory (kB)'); canvas2.draw()
    if passes: combo_mem.bind('<<ComboboxSelected>>', lambda e: (redraw(fig2, mem_df[mem_df['pass']==int(combo_mem.get())],'n','rss_kB',f'Memory vs n (pass {combo_mem.get()})','n','Memory (kB)'), canvas2.draw()))

    # Avg by n Tab
    tab3 = ttk.Frame(notebook); notebook.add(tab3, text='Avg by n')
    fig3 = Figure(figsize=(8,4), dpi=dpi); canvas3 = FigureCanvasTkAgg(fig3, master=tab3); canvas3.get_tk_widget().pack(fill='both', expand=True); NavigationToolbar2Tk(canvas3, tab3)
    fig3.clf(); ax1=fig3.add_subplot(121); ax2=fig3.add_subplot(122)
    if not stn.empty:
        stn_p = stn.pivot(index='n', columns='algorithm', values='mean')
        stn_p.plot(kind='bar', ax=ax1)
    ax1.set_title('Avg Time vs n'); ax1.set_xlabel('n'); ax1.set_ylabel('Time (s)')
    if not smn.empty:
        smn_p = smn.pivot(index='n', columns='algorithm', values='mean')
        smn_p.plot(kind='bar', ax=ax2)
    ax2.set_title('Avg Mem vs n'); ax2.set_xlabel('n'); ax2.set_ylabel('Mem (kB)')
    fig3.tight_layout(); canvas3.draw()

    # Stats Tab: use stats per n (across passes)
    tab4 = ttk.Frame(notebook); notebook.add(tab4, text='Stats')
    text = tk.Text(tab4)
    lines = []
    lines.append('Execution time (mean ± std-dev) across passes:')
    for _,row in stn.iterrows(): lines.append(f"n={row['n']} / {row['algorithm']}: {row['mean']:.6f} ± {row['std']:.6f} s")
    lines.append('\nPeak RSS (mean ± std-dev) across passes:')
    for _,row in smn.iterrows(): lines.append(f"n={row['n']} / {row['algorithm']}: {row['mean']:.0f} ± {row['std']:.0f} kB")
    if not corr.empty:
        lines.append('\nCorrectness flag (element-wise equality vs standard):')
        for _,r in corr.iterrows(): lines.append(f"pass {r.get('pass','-')} n={r.get('n','-')}: {r['correct']}")
    text.insert('1.0','\n'.join(lines))
    text.pack(fill='both', expand=True)

    # Table Tab
    tab5 = ttk.Frame(notebook); notebook.add(tab5, text='Table')
    tv = ttk.Treeview(tab5); tv['columns'] = list(table.columns); tv['show'] = 'headings'
    for c in table.columns: tv.heading(c, text=c); tv.column(c, width=100, anchor='center')
    for _,r in table.iterrows(): tv.insert('', 'end', values=list(r.values))
    tv.pack(fill='both', expand=True)

    root.mainloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark JSON Visualizer")
    parser.add_argument('-f','--file', required=True, help="Input JSON file")
    parser.add_argument('-w','--window-size', type=int, default=100, help="Figure DPI")
    args = parser.parse_args()
    df = load_and_flatten(args.file)
    time_df, mem_df, stp, smp, stn, smn, corr, table = prepare_data(df)
    build_gui(time_df, mem_df, stp, smp, stn, smn, corr, table, dpi=args.window_size)
    