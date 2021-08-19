import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, ScalarFormatter, NullFormatter, FixedLocator, NullLocator
from pandas import read_csv

DPI = 300
order = [
    'LU', 'CG + AMG', 'CG + GMG V-cycle', 'CG + full GMG',
    'Matfree CG + telescoped full GMG'
]
colors = {key: f'C{ii}' for ii, key in enumerate(order)}


## Singlecore
fig1, ax1 = plt.subplots(1, 2)
fig1.set_size_inches(12, 4)
single_raw = read_csv('singlecore.csv')
single = single_raw.pivot(index='dofs', columns='solver name', values='runtime')[order]
select_colors = [colors[k] for k in single.columns]
# Normal
single.plot(color=select_colors, legend=False, ax=ax1[0])
ax1[0].set_ylim([0, 220])
ax1[0].set_ylabel('runtime (s)')
f = ScalarFormatter()
f.set_scientific(True)
ax1[0].xaxis.set_major_formatter(f)
# Log
single.plot(logx=True, logy=True, color=select_colors, legend=False, ax=ax1[1])
ax1[1].yaxis.set_major_formatter(ScalarFormatter())
# Comparison lines
mins = single_raw.groupby('dofs').min()
xs = [mins.index[0], mins.index[-1]]
# h
min_start_val = 0.75*mins['runtime'].loc[xs[0]]
min_end_val = min_start_val*xs[1]/xs[0]
ymins = [min_start_val, min_end_val]
ax1[1].plot(xs, ymins, 'k--')
ax1[1].annotate(r'$O(n)$', xy=(0.4, 0.2), xycoords='axes fraction')
# h**5/3
xs = [mins.index[0], mins.index[-2]]
maxs = single_raw.groupby('dofs').max()
max_start_val = 1.25*maxs['runtime'].loc[xs[0]]
max_end_val = max_start_val*(xs[1]/xs[0])**(5/3)
ymaxs = [max_start_val, max_end_val]
ax1[1].plot(xs, ymaxs, 'k--')
ax1[1].annotate(r'$O(n^{5/3})$', xy=(0.1, 0.4), xycoords='axes fraction')

ax1[0].legend(title='solver name', labels=single.columns,
    bbox_to_anchor=(0, 1.02, 2.2, -0.2), loc='lower left',
    ncol=3, mode="expand", borderaxespad=0.)
fig1.suptitle('Computational cost of solvers, single core on ARCHER2', y=1.15)
fig1.savefig('single.png', bbox_inches='tight', dpi=DPI)

## Weak scaling
fig2, ax2 = plt.subplots(1, 1)
fig2.set_size_inches(12, 5)
weak_raw = read_csv('weak.csv')
weak = weak_raw.pivot(index='cores', columns='solver name', values='runtime')[order]
select_colors = [colors[k] for k in weak.columns]
weak.plot(logx=True, color=select_colors, ax=ax2)
ax2.set_xscale('log', base=2)

ax2.set_ylim([0, 80])
ax2.set_ylabel('runtime (s)')

annotate = weak_raw.groupby('cores').min()
ticks = annotate.index
ax2.xaxis.set_major_locator(FixedLocator(ticks))
ax2.xaxis.set_major_formatter(FixedFormatter(ticks))
ax2.xaxis.set_minor_formatter(NullFormatter())
for x, y, f in annotate[['runtime', 'dofs_core']].reset_index().values:
    anchor = 'center'
    offset = (0, -10)
    ax2.annotate(str(round(f)),
        xy=(x, y), xycoords='data',
        xytext=offset, textcoords='offset points',
        fontsize='medium', color='k', ha=anchor)
ax2.annotate('DOFs per core',
        xy=(0.5, 0.1), xycoords='axes fraction',
        xytext=(0, -25), textcoords='offset points',
        fontsize='medium', color='k', ha=anchor)
ax2.set_title('Solver weak scaling on one node of ARCHER2')
fig2.savefig('weak.png', bbox_inches='tight', dpi=DPI)

## Strong scaling
fig3, ax3 = plt.subplots(1, 1)
fig3.set_size_inches(12, 5)
strong_raw = read_csv('strong.csv')
order = [x for x in order if x in strong_raw['solver name'].unique()]
strong = strong_raw.pivot(index='cores', columns='solver name', values='runtime')[order]
select_colors = [colors[k] for k in strong.columns]
strong.plot(logx=True, logy=True, color=select_colors, ax=ax3)
ax3.set_xscale('log', base=2)
ax3.xaxis.set_major_formatter(ScalarFormatter())
ax3.xaxis.set_minor_formatter(ScalarFormatter())
ax3.yaxis.set_major_formatter(ScalarFormatter())
ax3.yaxis.set_minor_formatter(ScalarFormatter())
ax3.set_ylabel('runtime (s)')

# Secondary axis for nodes
annotate = strong_raw.groupby('cores').min()
cpn = 128  # Cores per node
secax = ax3.secondary_xaxis('top', functions=(lambda x: x/cpn, lambda y: cpn*y))
secax.set_xlabel('nodes', labelpad=-10)
ticks = [round(x/cpn) for x in annotate.index]
secax.xaxis.set_major_locator(FixedLocator(ticks))
secax.xaxis.set_major_formatter(FixedFormatter(ticks))
secax.xaxis.set_minor_locator(NullLocator())
secax.xaxis.set_minor_formatter(NullFormatter())

# Perfect scaling
xs = [annotate.index[0], annotate.index[-1]]
start_val = strong.loc[xs[0]].min() + 20
end_val = start_val*xs[0]/xs[1]
ys = [start_val, end_val]
h = ax3.plot(xs, ys, 'k--')
custom_lines = []
custom_lines.append(Line2D([0], [0], color='black', linestyle='--', lw=1))
custom = plt.legend(custom_lines, ['Perfect scaling'], loc='upper center')
ax3.add_artist(custom)
ax3.legend(title='solver name', labels=strong.columns)

# Annotations for DOFs per core
for x, y, f in annotate[['runtime', 'dofs_core']].reset_index().values:
    anchor = 'center'
    offset = (0, -10)
    ax3.annotate(str(round(f)),
        xy=(x, y), xycoords='data',
        xytext=offset, textcoords='offset points',
        fontsize='medium', color='k', ha=anchor)
ax3.annotate('DOFs per core',
        xy=(0.5, 0.25), xycoords='axes fraction',
        xytext=(0, -25), textcoords='offset points',
        fontsize='medium', color='k', ha=anchor)
ax3.set_title('Solver strong scaling across 8 nodes of ARCHER2')
fig3.savefig('strong.png', bbox_inches='tight', dpi=DPI)
