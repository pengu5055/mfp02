import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit


def rand_walk(n, a, m, speed):  # n; number of steps, speed; is constant speed
    x_prev = 0
    x_out = [0]
    y_prev = 0
    y_out = [0]
    t_prev = 0
    t = []
    oddaljenost = []
    for i in range(0, n):
        l = (np.random.pareto(a) + 1) * m
        phi = np.random.uniform() * 2 * np.pi
        print("Step: {}, L: {}, Phi: {}".format(i + 1, l, phi))
        x_prev += l * np.cos(phi)
        y_prev += l * np.sin(phi)
        x_out.append(x_prev)
        y_out.append(y_prev)
        oddaljenost.append(np.sqrt(x_prev**2 + y_prev**2))
        t_prev += l*speed
        t.append(t_prev)

    return np.array(x_out), np.array(y_out), np.array(t), np.array(oddaljenost)


def rand_flight(n, a, m, interval):  # n; number of steps, interval; time needed for step
    x_prev = 0
    x_out = [0]
    y_prev = 0
    y_out = [0]
    t_prev = 0
    t = []
    oddaljenost = []
    for i in range(0, n):
        l = (np.random.pareto(a) + 1) * m
        phi = np.random.uniform() * 2 * np.pi
        print("Step: {}, L: {}, Phi: {}".format(i + 1, l, phi))
        x_prev += l * np.cos(phi)
        y_prev += l * np.sin(phi)
        x_out.append(x_prev)
        y_out.append(y_prev)
        oddaljenost.append(np.sqrt(x_prev ** 2 + y_prev ** 2))
        t_prev += interval
        t.append(t_prev)

    return np.array(x_out), np.array(y_out), np.array(t), np.array(oddaljenost)


def linear_interpolation(x, x0, y0, x1, y1):
    return y0 + (x - x0)(y1 - y0)/(x1 - x0)


def run_tries(tries, function, *args):  # Returns array[run][dataset][element_of_data]
    return np.array([function(*args) for i in range(0, tries)])


def draw_tries(array, loc, x, y):  # Takes result array from run_tries and indicies x and y
    for element in array:
        loc.plot(element[x], element[y])


def sigma_tries_walk(array, t_end, t_start=1):  # Ce je t_start=0 imam probleme z log potem
    # Takes result array from run_tries and returns times and interpolated values
    tries = np.shape(array)[0]  # Wonky way of determining tries count
    steps = np.shape(array[0][0])[0] - 1  # Wonky way of determining step count which is constant between tries
    # times = np.arange(t_start, t_end, t_step)  # Interpoliram kar na case prvega ali je bolje na nek neodvisen range
    times = np.linspace(t_start, t_end, steps)  # Enakomerno razdeljeno?
    interpolated = []
    for i in range(0, tries):  # Change real data with interpolated data?
        dist = array[i][3]  # Distance array
        time = array[i][2]  # Time array
        interp = np.interp(times, time, dist)
        # interp = np.insert(interp, 0, 0)  # Vsili niclo na zacetek interp zato da dimenzije stimajo za dalje
        interpolated.append(interp)
        array[i][3] = interp

    return times, sigma_tries_flight(array)


def sigma_tries_flight(array):  # Takes result array from run_tries and returns 2 arrays
    sigma = []
    tries = np.shape(array)[0]  # Wonky way of determining tries count
    steps = np.shape(array[0][0])[0] - 1  # Wonky way of determining step count which is constant between tries
    for k in range(0, steps):
        dist = [array[i][3][k] for i in range(0, tries - 1)]
        # print(dist)
        sigma.append((st.median_abs_deviation(dist)/0.67449)**2)

    return np.array(sigma)


def gamma_data_flight(n, m, mu_start, mu_end, mu_step, scale=1, interval=1):
    # n number of iterations; m number of steps per iteration
    gamma = []
    mu = []
    a_start = mu_start - 1
    a_end = mu_end -1
    for a in np.arange(a_start, a_end, mu_step):
        data = run_tries(n, rand_flight, m, a, scale, interval)
        times = np.log(data[0][2])  # Bi se dalo posplositi
        sigma_eval = np.log(sigma_tries_flight(data))
        fitpar, fitcov = curve_fit(LinearFit, xdata=times, ydata=sigma_eval)
        gamma.append(fitpar[0])
        mu.append(a + 1)  # +1 zaradi definicije pareto

    return np.array(mu), np.array(gamma)


def gamma_data_walk(n, m, t_end, mu_start, mu_end, mu_step, scale=1, speed=1):
    # n number of iterations; m number of steps per iteration
    gamma = []
    mu = []
    a_start = mu_start - 1
    a_end = mu_end -1
    for a in np.arange(a_start, a_end, mu_step):
        data = run_tries(n, rand_walk, m, a, scale, speed)
        times, sigma_eval = np.log(sigma_tries_walk(data, t_end))
        print(sigma_eval)
        fitpar, fitcov = curve_fit(LinearFit, xdata=times, ydata=sigma_eval)
        gamma.append(fitpar[0])
        mu.append(a + 1)  # +1 zaradi definicije pareto

    return np.array(mu), np.array(gamma)

def pareto(a, m, x):
    return a * m ** a / x ** a


def LinearFit(x, k, n):
    return k*x + n


# Plot walks
# fig, (ax1, ax2) = plt.subplots(1, 2)
# x_data, y_data, time, oddaljenost = rand_walk(10, 1.5, 1, 1)
# x_data2, y_data2, time2, oddaljenost = rand_flight(100, 1.5, 1, 1)
# ax1.scatter(0, 0, c="#a41623", label="Izhodisce")
# ax1.plot(x_data, y_data, lw=1, c="#e07be0")
# ax1.scatter(x_data, y_data, s=5, c="#932f6d")
# ax2.scatter(0, 0, c="#a41623", label="Izhodisce")
# ax2.plot(x_data2, y_data2, lw=1, c="#e07be0")
# ax2.scatter(x_data2, y_data2, s=5, c="#932f6d")
# ax1.set_title(r"Sprehod pri $\mu = 2.5$ z $m = 10$")
# ax2.set_title(r"Sprehod pri $\mu = 2.5$ z $m = 100$")
# ax1.set_xlabel(r"$x$")
# ax1.set_ylabel(r"$y$")
# ax2.set_xlabel(r"$x$")
# ax2.set_ylabel(r"$y$")
# ax1.legend()
# ax2.legend()
# plt.show()

# Plot multiple tries
# fig, (ax1, ax2) = plt.subplots(1, 2)
# data = run_tries(10, rand_flight, 500, 1.5, 1, 1)
# draw_tries(data, ax1, 0, 1)
# draw_tries(data, ax2, 2, 3)
# ax1.set_xlabel(r"$x$")
# ax1.set_ylabel(r"$y$")
# ax2.set_xlabel(r"$t$")
# ax2.set_ylabel(r"$r$")
# plt.suptitle(r"Primer nekaj poletov in njihovih $r(t)$ za $\mu = 2.5$ in $m=500$")
# plt.show()

# Plot linear fit
# fig, ax = plt.subplots()
# data = run_tries(1000, rand_walk, 500, 1.75, 1, 1)
# Gledamo log log skalo zato da lahko sam fitam gor premico?
# times, sigma_eval = np.log(sigma_tries_walk(data, 1000))
#
# fitpar, fitcov = curve_fit(LinearFit, xdata=times, ydata=sigma_eval)
# yfit = LinearFit(times, fitpar[0], fitpar[1])
# fittext= "Linear fit: $y = kx + n$\nk = {} ± {}\nn = {} ± {}".format(format(fitpar[0], ".4e"), format(fitcov[0][0]**0.5, ".4e"),
#                                                                      format(fitpar[1], ".4e"), format(fitcov[1][1]**0.5, ".4e"))
# plt.text(0.5, 0.12, fittext, ha="left", va="center", size=10, transform=ax.transAxes, bbox=dict(facecolor="#a9f5ee", alpha=0.5))
# plt.plot(times, sigma_eval, c="#531253")
# plt.plot(times, yfit, c="#97D8B2")
# plt.xlabel(r"$\log{(t)}$")
# plt.ylabel(r"$\log{(\sigma^2)}$")
# plt.title(r"Prileganje premice pri $m=1000$, $n=500$, $\mu=2.75$ in $t_{\mathrm{end}}=1000$")
# plt.show()

# gamma(mu) plot for flights

bigdata = gamma_data_flight(100, 500, 1.1, 5, 0.1)
plt.axhline(y=1, color='#b0c592', linestyle=':', label=r"$\gamma =1$")
plt.title(r"$\gamma(\mu)$ pri n = {} in m = {}".format(100, 500))
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\gamma$")
plt.scatter(bigdata[0], bigdata[1], c="#b9314f", s=3)
plt.plot(bigdata[0], bigdata[1], c="#b298dc", label=r"$\gamma(\mu)$")
plt.legend()
plt.show()

# gamma(mu) plot for walks

# bigdata = gamma_data_walk(100, 500, 1000, 1.1, 5, 0.1)
# plt.axhline(y=1, color='#b0c592', linestyle=':', label=r"$\gamma =1$")
# plt.title(r"$\gamma(\mu)$ pri n = {}, m = {} in t_end = {}".format(100, 500, 1000))
# plt.xlabel(r"$\mu$")
# plt.ylabel(r"$\gamma$")
# plt.scatter(bigdata[0], bigdata[1], c="#b9314f", s=3)
# plt.plot(bigdata[0], bigdata[1], c="#b298dc", label=r"$\gamma(\mu)$")
# plt.legend()
# plt.show()
