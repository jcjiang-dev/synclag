import numpy as np
import matplotlib.pyplot as plt

def calculate_tau_m(a_thin, a_thick):
	return 3.0/2.0*(np.sqrt(1-8.0*a_thin/3.0/a_thick)-1.0)
'''
################# These are synchrotron emission that considers more realistic ones as in Turler 2000 ##############
def calculate_s_mu(mu, mu_m, a_thin, a_thick, mu_h, mu_b):
	t_m = calculate_tau_m(a_thin, a_thick)
	if mu < mu_h:
		a_thick=5.0/2.0
	if mu > mu_b:
		a_thin=a_thin-0.5
	temp1 = np.power(mu/mu_m, a_thick)
	temp2 = -1.0* t_m* np.power(mu/mu_m, a_thin-a_thick)
	temp3 = 1.0-np.exp(temp2)
	temp4 = 1.0-np.exp(-1.0* t_m)
	return s_m* temp1 * temp3 / temp4
def calculate_spec_archive(s_m* mu, mu_m, a_thin, a_thick, mu_h, mu_b):
	s_mu=[]
	for i in range(len(mu)):
		s_mu.append(calculate_s_mu(mu[i], mu_m, a_thin, a_thick, mu_h, mu_b))

	s_mu = np.array(s_mu)
	idx_h = np.argmin(np.abs(mu - mu_h))
	idx_b = np.argmin(np.abs(mu - mu_b))

	# Scale the left side of mu_h and mu_b
	s_mu[:idx_h] *= s_mu[idx_h] / s_mu[idx_h - 1]
	s_mu[idx_b:] *= s_mu[idx_b-1] / s_mu[idx_b]
	
	return s_m*s_mu
'''

def calculate_s_mu(s_m, mu, mu_m, a_thin, a_thick):
	t_m = calculate_tau_m(a_thin, a_thick)
	temp1 = np.power(mu/mu_m, a_thick)
	temp2 = -1.0* t_m* np.power(mu/mu_m, a_thin-a_thick)
	temp3 = 1.0-np.exp(temp2)
	temp4 = 1.0-np.exp(-1.0* t_m)
	return s_m* temp1 * temp3 / temp4
def calculate_spec(s_m, mu, mu_m, a_thin, a_thick):
	s_mu=[]
	for i in range(len(mu)):
		s_mu.append(calculate_s_mu(s_m, mu[i], mu_m, a_thin, a_thick))
	return s_mu

def double_bkn_pl(t, t_r, t_p, index_1, index_2, index_3, A):
	# Compute normalization constants to ensure continuity
	A1 = A / (t_r**index_1)  # Normalize so that y(t_r) = A
	A2 = A1 * (t_r**(index_1 - index_2))  # Ensures continuity at t_r
	A3 = A2 * (t_p**(index_2 - index_3))  # Ensures continuity at t_p	
	# Compute the power-law function piecewise
	y = np.piecewise(t, [t < t_r, (t >= t_r) & (t < t_p), t >= t_p], [lambda t: A1 * t**index_1, lambda t: A2 * t**index_2, lambda t: A3 * t**index_3])
	return y

def log_linear_func(t, t_r, t_p, y1, y2):
	y = np.piecewise(t, [t < t_r, t > t_p, (t >= t_r) & (t <= t_p)], [y1, y2, lambda t: y1 + (y2 - y1) * (np.log10(t) - np.log10(t_r)) / (np.log10(t_p) - np.log10(t_r))])
	return y

def calculate_lc(t, mu_ref, tr, tp, mu_m_tr, s_m_tr, alpha_thin_tr, alpha_thin_tp, alpha_thick_tr, alpha_thick_tp, gamma1, gamma2, gamma3, beta1, beta2, beta3):
	a_thin=log_linear_func(t, tr, tp, alpha_thin_tr, alpha_thin_tp)
	a_thick=log_linear_func(t, tr, tp, alpha_thick_tr, alpha_thick_tp)
	s_m=double_bkn_pl(t, tr, tp, gamma1, gamma2, gamma3, s_m_tr)
	mu_m=double_bkn_pl(t, tr, tp, beta1, beta2, beta3, mu_m_tr)
	flux=[]
	for i in range(len(t)):
		flux.append( calculate_s_mu(s_m[i], mu_ref, mu_m[i], a_thin[i], a_thick[i]) )
	return flux

nd=365.0
tr=0.14
tp=1.63
mu_m_tr=1.20e11
s_m_tr=15.30
alpha_thin_tr=-1.09
alpha_thin_tp=-0.48
alpha_thick_tr=1.55
alpha_thick_tp=1.74
beta1=-0.51
beta2=-0.88
beta3=-1.19
gamma1=0.51
gamma2=0.02
gamma3=-1.36

t=np.linspace(0.01,40.0,4000)
mu_ref1=2.5e9
mu_ref2=8.0e9
mu_ref3=37.0e9
s1=calculate_lc(t, mu_ref1, tr, tp, mu_m_tr, s_m_tr, alpha_thin_tr, alpha_thin_tp, alpha_thick_tr, alpha_thick_tp, gamma1, gamma2, gamma3, beta1, beta2, beta3)
s2=calculate_lc(t, mu_ref2, tr, tp, mu_m_tr, s_m_tr, alpha_thin_tr, alpha_thin_tp, alpha_thick_tr, alpha_thick_tp, gamma1, gamma2, gamma3, beta1, beta2, beta3)
s3=calculate_lc(t, mu_ref3, tr, tp, mu_m_tr, s_m_tr, alpha_thin_tr, alpha_thin_tp, alpha_thick_tr, alpha_thick_tp, gamma1, gamma2, gamma3, beta1, beta2, beta3)

# Make an ugly figure of the light curves
plt.plot(t,s1,label="%.1e" %mu_ref1)
plt.plot(t,s2,label="%.1e" %mu_ref2)
plt.plot(t,s3,label="%.1e" %mu_ref3)
plt.xlim(0.0,20.0)
plt.xscale("linear")
plt.yscale("linear")
plt.xlabel("t (years)")
plt.ylabel("$S_\mu$")
plt.legend()
plt.title("Synchrotron LC")
plt.grid(True, which="both", ls="--")
plt.show()

def compute_time_lag(t, s1, s2):
    # Compute time step (assuming uniform spacing)
    dt = t[1] - t[0]
    S1 = np.fft.rfft(s1)  # Real-to-complex FFT
    S2 = np.fft.rfft(s2)
    cross_spectrum = S1 * np.conj(S2)  # S1 * conjugate of S2
    phase_lag = np.angle(cross_spectrum)  # Extract phase
    freq = np.fft.rfftfreq(len(t), d=dt)  # Frequencies corresponding to FFT bins
    time_lag = np.zeros_like(phase_lag)
    nonzero_freq = freq > 0  # Avoid division by zero at f = 0
    time_lag[nonzero_freq] = phase_lag[nonzero_freq] / (2 * np.pi * freq[nonzero_freq])
    return freq, time_lag


freq, time_lag = compute_time_lag(t, s1, s2) #### 8GHz (mu_ref2)
# Plot Phase Lag vs. Fourier Frequency
plt.figure(figsize=(8, 6))
plt.plot(freq/365.0, time_lag*365.0, marker="o", linestyle="None", color="b")
plt.xlabel("Fourier Frequency (Hz)")
plt.ylabel("Time Lag (days)")
plt.title("Time Lag vs. Frequency")
plt.grid(True, which="both", linestyle="--")
plt.xscale("log")  # Log scale for better visualization if needed
plt.title("Lag relative of 2.5GHz relative to 8GHz")
plt.show()
