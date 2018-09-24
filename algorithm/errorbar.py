import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.5, 5.5, 0.5)

y = np.exp(-x)
xerr = 0.1
yerr = 0.2
# lower & upper limits of the error
lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
ls = 'dotted'
fig, ax = plt.subplots(figsize=(7, 4))
# standard error bars
ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)
# including upper limits
ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, linestyle=ls)
# including lower limits
ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, linestyle=ls)
# including lower limits and upper limits
ax.errorbar(x, y + 1.5, xerr=xerr, yerr=yerr, lolims=lolims, uplims=uplims, marker='o', markersize=8, linestyle=ls)
xerr = 0.2
yerr = np.zeros(x.shape) + 0.2
yerr[[3, 6]] = 0.3
# mock up some limits by modifying previous data
xlolims = lolims
xuplims = uplims
lolims = np.zeros(x.shape)
uplims = np.zeros(x.shape)
lolims[[6]] = True
uplims[[3]] = True
ax.errorbar(x, y + 2.1, xerr=xerr, yerr=yerr, xlolims=xlolims, xuplims=xuplims,
            uplims=uplims, lolims=lolims, marker='o', markersize=8, linestyle='none')
ax.set_xlim((0, 5.5))
ax.set_title('errorbar upper and lower limits')
plt.show()
