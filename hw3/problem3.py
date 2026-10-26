# Course: MAE 459 (001) - Rocket Propulsion
# Instructor: Prof. Braun
# Author: Jaey Kim (student)
# Date: 10/20/2026

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
P0 = 101325.0                 # sea-level pressure, Pa
Pc = 200.0 * P0              # chamber stagnation pressure, Pa
At = 0.08                    # sonic throat area, m^2
mdot = 550.0                 # total propellant mass flow, kg/s
R = 605.0                    # specific gas constant, J/(kg K)
gamma = 1.15
g0 = 9.80665                 # standard gravity, m/s^2
H = 8013.2                   # atmosphere scale height, m
hd = 6700.0                  # ideal-expansion design altitude, m
k = 2.75                    # course Summerfield separation factor

# 1. Design pressure: ideal expansion ONLY at the design altitude.
Pe = P0 * np.exp(-hd/H)
# 2. Invert the isentropic pressure relation to obtain exit Mach number.
Me = np.sqrt(2/(gamma-1) * ((Pc/Pe)**((gamma-1)/gamma)-1))
# 3. Area-Mach relation: fixes the geometry for ALL operating altitudes.
epsilon = (1/Me) * (2/(gamma+1)*(1+(gamma-1)*Me**2/2))**(
    (gamma+1)/(2*(gamma-1)))
Ae = epsilon * At
# 4. Separation begins when Pa > k*Pe. Check the largest Pa (sea level).
hsep = hd - H*np.log(k)
assert P0 <= k*Pe, 'Separation correction needed: this attached-flow model is not valid.'
# 5. Momentum coefficient and vacuum thrust coefficient.
CFmomentum = np.sqrt(2*gamma**2/(gamma-1)*(2/(gamma+1))**(
    (gamma+1)/(gamma-1))*(1-(Pe/Pc)**((gamma-1)/gamma)))
CFvac = CFmomentum + epsilon*Pe/Pc
cstar = Pc*At/mdot

def performance(altitude_m):
    """Return ambient pressure (Pa), CF, thrust (N), and Isp (s)."""
    Pa = P0*np.exp(-np.asarray(altitude_m)/H)
    CF = CFmomentum + epsilon*(Pe-Pa)/Pc
    F = CF*Pc*At
    return Pa, CF, F, F/(mdot*g0)

# Independent check using inferred chamber temperature and exhaust momentum.
Tc = cstar**2*gamma/R*(2/(gamma+1))**((gamma+1)/(gamma-1))
Te = Tc/(1+(gamma-1)*Me**2/2)
ue = Me*np.sqrt(gamma*R*Te)
h = np.linspace(0, 100000, 1001)
Pa, CF, F, Isp = performance(h)
assert np.all(Pa <= k*Pe)
assert np.allclose(F, mdot*ue+(Pe-Pa)*Ae)
assert np.isclose(performance(hd)[1], CFmomentum)

lines = []
def report(s=''):
    lines.append(s)
report('MAE 459 | HOMEWORK 3, PROBLEM 3')
report('Fixed nozzle; ideal-gas, constant-gamma, isentropic attached flow.')
report('All pressures absolute. Altitude h is in meters in all equations.')
report('\nGIVEN')
report(f'Pc = 200 atm x 101325 Pa/atm = {Pc:,.0f} Pa')
report(f'mdot = {mdot:g} kg/s; At = {At:g} m^2; gamma = {gamma}; R = {R:g} J/(kg K)')
report(f'h_design = {hd:g} m; H = {H:g} m; g0 = {g0} m/s^2; k = {k}')
report('\n1. DESIGN EXIT PRESSURE')
report(f'Pa(h) = 101325 exp(-h/8013.2) Pa')
report(f'Pe = Pa(6700) = 101325 exp(-6700/8013.2) = {Pe:,.6f} Pa')
report('\n2. EXIT MACH NUMBER')
report('Me = sqrt{2/(gamma-1) [(Pc/Pe)^((gamma-1)/gamma) - 1]}')
report(f'   = sqrt{{2/0.15 [({Pc:.0f}/{Pe:.6f})^(0.15/1.15) - 1]}}')
report(f'Me = {Me:.8f}')
report('\n3. FIXED NOZZLE AREA')
report('epsilon = (1/Me) [2/(gamma+1) (1+(gamma-1)Me^2/2)]^((gamma+1)/(2(gamma-1)))')
report(f'epsilon = {epsilon:.8f}; Ae = epsilon x 0.08 = {Ae:.8f} m^2')
report('\n4. SUMMERFIELD CHECK')
report(f'Separation condition: Pa > 2.75 Pe = {k*Pe:,.3f} Pa')
report(f'Maximum Pa = {P0:,.0f} Pa < {k*Pe:,.3f} Pa: no separation in 0-100 km.')
report(f'h_sep = 6700 - 8013.2 ln(2.75) = {hsep:.3f} m (outside requested range).')
report('Keep Pe and Ae fixed. Do not apply Pe_eff = Pa/2.75 in this range.')
report('\n5. THRUST COEFFICIENT')
report('CF_momentum = sqrt{[2 gamma^2/(gamma-1)] [2/(gamma+1)]^((gamma+1)/(gamma-1))')
report('                   x [1-(Pe/Pc)^((gamma-1)/gamma)]}')
report(f'CF_momentum = {CFmomentum:.9f}')
report(f'CF_vac = CF_momentum + epsilon Pe/Pc = {CFvac:.9f}')
report(f'CF(h) = {CFvac:.9f} - {epsilon*P0/Pc:.9f} exp(-h/{H})')
report('\n6. THRUST AND SPECIFIC IMPULSE')
report(f'F(h) = Pc At CF(h) = {Pc*At:,.0f} CF(h) N')
report(f'Isp(h) = F(h)/(mdot g0) = F(h)/{mdot*g0:.6f} s')
report(f'Equivalent: c* = Pc At/mdot = {cstar:.6f} m/s; Isp = CF c*/g0.')
report('\nRESULTS')
report(f'{"h (m)":>10} {"Pa (Pa)":>13} {"CF":>12} {"F (N)":>16} {"Isp (s)":>12}')
for z in [0,6700,10000,20000,40000,60000,100000,np.inf]:
    p,cf,f,isp = performance(z)
    label = 'vacuum' if np.isinf(z) else f'{z:.0f}'
    report(f'{label:>10} {p:13.5f} {cf:12.7f} {f:16,.2f} {isp:12.4f}')
report('\nCHECKS AND INTERPRETATION')
report('Verified F = mdot ue + (Pe-Pa)Ae independently at every plotted altitude.')
report(f'Inferred ideal-model Tc = {Tc:.2f} K; ue = {ue:.3f} m/s.')
report('Tc is a consistency check, not a required input or a real-SSME measurement.')
report('Below 6.7 km: overexpanded but attached. At 6.7 km: ideally expanded.')
report('Above 6.7 km: underexpanded. Falling ambient pressure increases pressure thrust.')
report('100 km is approximately vacuum; the exponential model does not give exactly zero Pa.')
report('\nCourse references: Lecture 9 slides 7, 9, 13-14; Lecture 11 slides 37-38.')
text = '\n'.join(lines)+'\n'
print(text)
(OUT/'calculation.txt').write_text(text)

# Three uncluttered panels, with design-altitude and vacuum-limit guides.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(3,1,figsize=(8,8.5),sharex=True,layout='constrained')
fig.suptitle('Fixed-nozzle performance plot',fontsize=15,fontweight='bold')
for ax, values, limit, label in zip(axes,[CF,F,Isp],
    [CFvac,CFvac*Pc*At,CFvac*cstar/g0],
    ['Thrust coefficient, CF','Thrust (N)','Specific impulse (s)']):
    ax.plot(h/1000,values,color='#126782',linewidth=2.2)
    ax.axvline(hd/1000,color='#bc6c25',linestyle='--',linewidth=1,label='Design altitude: 6.7 km')
    ax.axhline(limit,color='#687782',linestyle=':',linewidth=1,label='Vacuum limit')
    ax.set_ylabel(label); ax.grid(alpha=.18); ax.set_xlim(0,100)
axes[0].legend(loc='lower right',fontsize=9,frameon=False)
axes[1].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
axes[2].set_xlabel('Altitude (km)')
fig.savefig(OUT/'plot.png',dpi=200)
plt.close(fig)
print('Saved calculation.txt and plot.png beside the script.')
