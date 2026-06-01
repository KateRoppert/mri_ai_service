#!/usr/bin/env python3
"""Regenerate MAS architecture figure — clean readable redesign."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIG_PATH = "/home/ubuntu/mri_ai_service/docs/presentation/figures/fig_mas_arch.png"

# ── Colors (matching presentation palette) ────────────────────────────────────
C_DARK  = "#008F4A"
C_MID   = "#00AC5A"
C_LITE  = "#E8F5ED"
C_BK    = "#1A1A1A"
C_GRAY  = "#666666"
C_LGRAY = "#F0F0F0"
C_AMBER = "#F57C00"  # highlight for bottleneck stage

# Stage type colours  (fill, border, text)
T_IO   = ("#C8EAD3", "#1B6B38", "#1B6B38")
T_CPU  = ("#BBDAF0", "#1C4B7C", "#1C4B7C")
T_CPUH = ("#DDD4F5", "#3D2E8F", "#3D2E8F")   # S5 – bottleneck
T_GPU  = ("#F7D4C8", "#8B2500", "#8B2500")

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8.2), dpi=150, facecolor='white')

# Main diagram occupies left 63 %, BDI+CNP panel right 37 %
ax  = fig.add_axes([0.01, 0.01, 0.60, 0.98])   # architecture
ax2 = fig.add_axes([0.63, 0.01, 0.36, 0.98])   # info panel

for a in (ax, ax2):
    a.set_xlim(0, 10);  a.set_ylim(0, 10);  a.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rbox(a, cx, cy, w, h, fill, edge, text='', text2='', fs=13, fs2=11, fc='white'):
    """Rounded box with optional two-line label."""
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=fill, edgecolor=edge, linewidth=2.2, zorder=3)
    a.add_patch(p)
    if text:
        y_off = 0.15 if text2 else 0
        a.text(cx, cy + y_off, text,
               ha='center', va='center', fontsize=fs,
               fontweight='bold', color=fc, zorder=4)
    if text2:
        a.text(cx, cy - 0.3, text2,
               ha='center', va='center', fontsize=fs2,
               color=fc, zorder=4, style='italic')

def hline(a, x0, x1, y, color='#888', lw=1.8, ls='-'):
    a.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=2)

def vline(a, x, y0, y1, color='#888', lw=1.8):
    a.plot([x, x], [y0, y1], color=color, lw=lw, zorder=2)

def arrowdown(a, x, y_from, y_to, color='#555', both=False):
    """Downward arrow."""
    style = '<->' if both else '->'
    a.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=1.8, mutation_scale=16), zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — BROKER AGENT
# ─────────────────────────────────────────────────────────────────────────────
ax.text(5.0, 9.65, 'TIER 1',
        ha='center', va='center', fontsize=10, color=C_GRAY,
        fontweight='bold', style='italic')

rbox(ax, 5.0, 9.05, 9.0, 1.0,
     fill=C_DARK, edge=C_DARK,
     text='BROKER AGENT',
     text2='Resource Pool Manager  ·  Negotiation Mediator',
     fs=16, fs2=12, fc='white')

# ─────────────────────────────────────────────────────────────────────────────
# BUS: Broker → Stage Managers
# ─────────────────────────────────────────────────────────────────────────────
# Stage manager x positions
xs = [0.75, 2.5, 4.25, 6.0, 7.75, 9.25]

# Label
ax.text(5.0, 7.9, 'Contract Net Protocol  ( Announcement → Bids → Award )',
        ha='center', va='center', fontsize=11, color=C_GRAY, style='italic')

# Vertical from broker bottom to horizontal bus
vline(ax, 5.0, 8.55, 7.7, color='#666')
# Horizontal bus
hline(ax, xs[0], xs[-1], 7.7, color='#666', lw=2.0)
# Drops from bus to each agent top
for xi in xs:
    arrowdown(ax, xi, 7.7, 7.1, color='#555')

# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — BDI STAGE MANAGER AGENTS
# ─────────────────────────────────────────────────────────────────────────────
ax.text(5.0, 6.85, 'TIER 2  —  BDI Stage Manager Agents',
        ha='center', va='center', fontsize=10, color=C_GRAY,
        fontweight='bold', style='italic')

stages = [
    ('S1\nBIDS',     'sat = 8\n3.49×',  T_IO),
    ('S2\nMetadata', 'sat = 8\n3.49×',  T_IO),
    ('S3\nNIfTI',    'sat = 8\n3.79×',  T_IO),
    ('S4\nQuality',  'sat = 12\n5.22×', T_CPU),
    ('S5\nPreproc.', 'sat = 6\n2.17×',  T_CPUH),  # bottleneck
    ('S6\nSegment.', 'sat = 2 GPU\n1.94×', T_GPU),
]

for xi, (name, sat, (bg, eg, fg)) in zip(xs, stages):
    # S5 gets a special highlight as bottleneck
    extra_lw = 3.5 if 'Preproc' in name else 2.2
    p = FancyBboxPatch((xi - 0.95, 5.05), 1.9, 1.65,
                       boxstyle="round,pad=0.12",
                       facecolor=bg, edgecolor=eg,
                       linewidth=extra_lw, zorder=3)
    ax.add_patch(p)
    ax.text(xi, 6.12, name, ha='center', va='center',
            fontsize=13, fontweight='bold', color=fg, zorder=4)
    ax.text(xi, 5.42, sat, ha='center', va='center',
            fontsize=11.5, color=fg, zorder=4)
    # Bottleneck badge
    if 'Preproc' in name:
        ax.text(xi, 5.05, '⚠ bottleneck', ha='center', va='top',
                fontsize=10, color=C_AMBER, fontweight='bold', zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# BUS: Stage Managers → Workers
# ─────────────────────────────────────────────────────────────────────────────
hline(ax, xs[0], xs[-1], 4.75, color='#666', lw=2.0)
for xi in xs:
    vline(ax, xi, 5.05, 4.75, color='#666')

# Arrow label
ax.text(5.0, 4.35, 'allocate  /  release  workers',
        ha='center', va='center', fontsize=11, color=C_GRAY, style='italic')
arrowdown(ax, 5.0, 4.75, 4.0, color='#555')

# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — WORKER PROCESSES
# ─────────────────────────────────────────────────────────────────────────────
ax.text(5.0, 3.75, 'TIER 3  —  Worker Processes',
        ha='center', va='center', fontsize=10, color=C_GRAY,
        fontweight='bold', style='italic')

# CPU workers (S1–S5)
rbox(ax, 4.0, 3.05, 7.3, 1.1,
     fill=C_LGRAY, edge='#AAAAAA',
     text='CPU Worker Processes',
     text2='1–8 workers per stage  (Stages 1–5)',
     fs=14, fs2=12, fc=C_BK)

# GPU workers (S6)
rbox(ax, 9.1, 3.05, 1.6, 1.1,
     fill='#FFF0E8', edge='#8B2500',
     text='GPU\nWorkers',
     text2='1–2 GPUs\nStage 6',
     fs=13, fs2=11, fc='#8B2500')

# Tier label
ax.text(0.3, 3.65, 'Tier 3', fontsize=10, color=C_GRAY,
        fontweight='bold', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND — stage type colours at bottom
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    ('#C8EAD3', '#1B6B38', 'I/O-bound  (S1–S3)'),
    ('#BBDAF0', '#1C4B7C', 'CPU-bound  (S4)'),
    ('#DDD4F5', '#3D2E8F', 'CPU-heavy  (S5 ⚠ bottleneck)'),
    ('#F7D4C8', '#8B2500', 'GPU-bound  (S6)'),
]
lx = 0.3
for bg, eg, label in legend_items:
    p = FancyBboxPatch((lx, 1.65), 0.45, 0.35,
                       boxstyle="round,pad=0.05",
                       facecolor=bg, edgecolor=eg, linewidth=1.8)
    ax.add_patch(p)
    ax.text(lx + 0.6, 1.83, label,
            ha='left', va='center', fontsize=11, color=C_BK)
    lx += 2.45

# ─────────────────────────────────────────────────────────────────────────────
# INFO PANEL (ax2) — BDI Model + Contract Net
# ─────────────────────────────────────────────────────────────────────────────

def info_section(a, title, items, y_start, title_color=C_DARK):
    """items: list of (bullet, description) tuples."""
    a.text(5.0, y_start, title,
           ha='center', va='center', fontsize=14,
           fontweight='bold', color=title_color)
    a.plot([0.3, 9.7], [y_start - 0.3, y_start - 0.3],
           color=C_MID, lw=1.5)
    y = y_start - 0.75
    for bullet, desc in items:
        a.text(0.5, y, bullet, ha='left', va='top',
               fontsize=13, color=title_color, fontweight='bold')
        a.text(1.1, y - 0.35, desc, ha='left', va='top',
               fontsize=11.5, color=C_BK)
        y -= 1.35
    return y

# BDI Agent Model
p_bdi = FancyBboxPatch((0.2, 5.1), 9.6, 4.65,
                       boxstyle="round,pad=0.15",
                       facecolor=C_LITE, edgecolor=C_MID, linewidth=2.0)
ax2.add_patch(p_bdi)

info_section(ax2, 'BDI Agent Model', [
    ('▸ Beliefs',    'Saturation profiles from Table I\n(updated at runtime)'),
    ('▸ Desires',    'Minimize queue wait;\navoid allocation beyond saturation'),
    ('▸ Intentions', 'Request / release / maintain\nthe current worker allocation'),
], y_start=9.35)

# Contract Net Protocol
p_cnp = FancyBboxPatch((0.2, 0.15), 9.6, 4.7,
                       boxstyle="round,pad=0.15",
                       facecolor='#FAFFF8', edgecolor=C_DARK, linewidth=2.0)
ax2.add_patch(p_cnp)

info_section(ax2, 'Contract Net Protocol', [
    ('① Announce',   'Stage Manager → Broker:\n"I need more workers"'),
    ('② Bid',        'Broker → all Stage Managers:\n"Who can release workers?"'),
    ('③ Award',      'Broker assigns; agents may refuse\nif local queue is non-empty'),
], y_start=4.5)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {FIG_PATH}")
