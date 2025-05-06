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
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Flatten nested JSON to columns like standard.time_s, blocked.rss_kB, etc.
    df = pd.json_normalize(data['results'])
    df.columns = [c.replace('.', '_') for c in df.columns]
    return df

def prepare_data(df):
    # Melt time and memory separately
    time_df = df.melt(id_vars=['pass','n'],
                      value_vars=['standard_time_s','blocked_time_s','strassen_time_s'],
                      var_name='algorithm', value_name='time_s')
    mem_df = df.melt(id_vars=['pass','n'],
                     value_vars=['standard_rss_kB','blocked_rss_kB','strassen_rss_kB'],
                     var_name='algorithm', value_name='rss_kB')
    for d in (time_df, mem_df):
        d['algorithm'] = d['algorithm'].str.replace(r'_(time_s|rss_kB)$','',regex=True)

    # Averages per pass and per n
    avg_time_pass = time_df.groupby(['pass','algorithm'])['time_s'].mean().reset_index()
    avg_mem_pass  = mem_df.groupby(['pass','algorithm'])['rss_kB'].mean().reset_index()
    avg_time_n = time_df.groupby(['n','algorithm'])['time_s'].mean().reset_index()
    avg_mem_n  = mem_df.groupby(['n','algorithm'])['rss_kB'].mean().reset_index()
    return time_df, mem_df, avg_time_pass, avg_mem_pass, avg_time_n, avg_mem_n

def build_gui(time_df, mem_df, avg_tp, avg_mp, avg_tn, avg_mn, dpi):
    root = tk.Tk()
    root.wm_title("Benchmark Visualizer for Matrix Multiplication Algorithms")
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)
    passes = sorted(time_df['pass'].unique())

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
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='Time (s)')
    combo_time = ttk.Combobox(tab1, values=passes, state='readonly')
    combo_time.set(passes[0]); combo_time.pack(anchor='nw', pady=5, padx=5)
    fig_time = Figure(figsize=(6,4), dpi=dpi)
    canvas_time = FigureCanvasTkAgg(fig_time, master=tab1)
    canvas_time.get_tk_widget().pack(fill='both', expand=True)
    NavigationToolbar2Tk(canvas_time, tab1)
    redraw(fig_time, time_df[time_df['pass']==passes[0]], 'n','time_s', f'Time vs 2^[n] Matrix Size (pass {passes[0]})','n','Time (s)')
    canvas_time.draw()
    combo_time.bind('<<ComboboxSelected>>', lambda e: (redraw(fig_time, time_df[time_df['pass']==int(combo_time.get())], 'n','time_s', f'Time vs n (pass {combo_time.get()})','n','Time (s)'), canvas_time.draw()))

    # Memory Tab
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='Memory (kB)')
    combo_mem = ttk.Combobox(tab2, values=passes, state='readonly')
    combo_mem.set(passes[0]); combo_mem.pack(anchor='nw', pady=5, padx=5)
    fig_mem = Figure(figsize=(6,4), dpi=dpi)
    canvas_mem = FigureCanvasTkAgg(fig_mem, master=tab2)
    canvas_mem.get_tk_widget().pack(fill='both', expand=True)
    NavigationToolbar2Tk(canvas_mem, tab2)
    redraw(fig_mem, mem_df[mem_df['pass']==passes[0]], 'n','rss_kB', f'Memory vs 2^[n] Matrix Size (pass {passes[0]})','n','Memory (kB)')
    canvas_mem.draw()
    combo_mem.bind('<<ComboboxSelected>>', lambda e: (redraw(fig_mem, mem_df[mem_df['pass']==int(combo_mem.get())], 'n','rss_kB', f'Memory vs n (pass {combo_mem.get()})','n','Memory (kB)'), canvas_mem.draw()))

    # Avg by n Tab: separate axes for time and memory
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text='Avg by n')
    fig_avg_n = Figure(figsize=(8,4), dpi=dpi)
    canvas_avg_n = FigureCanvasTkAgg(fig_avg_n, master=tab3)
    canvas_avg_n.get_tk_widget().pack(fill='both', expand=True)
    NavigationToolbar2Tk(canvas_avg_n, tab3)
    # clear and create two subplots
    fig_avg_n.clf()
    ax1 = fig_avg_n.add_subplot(121)
    avg_time_pivot = avg_tn.pivot(index='n', columns='algorithm', values='time_s')
    avg_time_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('Avg Time vs n Matrix Size')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Time (s)')
    ax2 = fig_avg_n.add_subplot(122)
    avg_mem_pivot = avg_mn.pivot(index='n', columns='algorithm', values='rss_kB')
    avg_mem_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('Avg Memory vs n Matrix Size')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Memory (kB)')
    fig_avg_n.tight_layout()
    canvas_avg_n.draw()

    # Statistics Tab
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text='Stats')
    stats_df = time_df.groupby(['algorithm','n'])['time_s'].describe()
    txt = tk.Text(tab4)
    txt.insert('1.0', stats_df.to_string())
    txt.pack(fill='both', expand=True)

    root.mainloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize benchmark JSON")
    parser.add_argument('-f','--file', required=True, help="Input JSON file (quote names with parentheses)")
    parser.add_argument('-w','--window-size', type=int, default=100, help="Figure DPI (default 100)")
    args = parser.parse_args()
    df = load_and_flatten(args.file)
    time_df, mem_df, avg_tp, avg_mp, avg_tn, avg_mn = prepare_data(df)
    build_gui(time_df, mem_df, avg_tp, avg_mp, avg_tn, avg_mn, dpi=args.window_size)
