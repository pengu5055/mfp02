import numpy as np
import matplotlib.pyplot as plt

#pareto distribution, pazi na (a+1) potenco!
a, m = 3., 2.  # shape and mode
s = (np.random.pareto(a, 1000) + 1) * m
print(s)
#Display the histogram of the samples, along with the probability density function:
plt.figure(figsize=(6,3))
count, binEdges, _ = plt.hist(s, 100, density=True,color='b',label="Sampled")
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
fit = a*m**a / bincenters**(a+1)
plt.plot(bincenters, max(count)*fit/max(fit), linewidth=2, color='r',label="Pareto")
plt.xlabel('x')
plt.ylabel('1/N * dN/dx')
#plt.yscale('log')
plt.legend()
plt.show()