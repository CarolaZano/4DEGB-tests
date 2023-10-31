#Calculate the conformal time in terms of x

import numpy as np
import cosmofuncsoptdep as cf
import scipy.integrate

# CAROLA: Cosmological parameters need to be changed as appropriate (e.g. at each step in chain) instead of fixed as here.
HH0,Nnu,wM0,wB0, A_s109, n_s=np.loadtxt('./UniverseVars.txt',unpack=True)


#Current values of energy densities
OmR0=2.47*10**(-5)/(HH0/100.)**2
OmN0=Nnu*(7./8.)*(4./11.)**(4./3.)*OmR0
OmM0=wM0/(HH0/100.)**2
OmB0=wB0/(HH0/100.)**2
OmL0=1.-OmM0-OmB0-OmR0-OmN0

#Small h, Hubble constant
h=HH0/100. #unitless
print('h=',h)

# Constants used to get optical depth
Delta2s1s=8.227 # units s^(-1)
eps=13.605698 # units eV
me=9.10938291*10**(-31) # units kg
T0=2.725 # units K
alpha=1./137. #fine structure
pi=3.14159265359
G=6.67428*10**(-11) #units m^3kg^(-1)s^(-2)
mH= 1.673575*10**(-27) # units kg 
sigmat=6.6524616*10**(-29) #Thomson cross section, units m^2

# Conversion factors
kb=8.6173324*10**(-5) # units eV
hb= 1.054571726*10**(-34)#6.626068*10**(-34) units kg m^2 s^(-1) 
c=2.99792458*10**(8) # units m/s
evJ=1.6*10**(-19) #units J/eV
MpCm=3.085678*10**22 #units m/MpC

H0=10**(5)/c # units h/MpC

# Stick everything in a vector
params=[Delta2s1s,eps,me,T0,alpha,kb,hb,c,evJ,MpCm,OmM0,OmR0,OmB0,OmL0, H0, OmN0]

############################# Set initial condtions and our grid for integration ####################################
	
def inttime(ai):
	return 1./(H0*(OmR0+OmN0+(OmM0+OmB0)*ai+OmL0*ai**4)**(0.5))
	
tautoday, err = scipy.integrate.quad(inttime,0.0,1.0)
	
#Get an initial conformal time for which a is sufficiently small 
for i in range(0,100):
	alim=10**(-7-i)
	ans, err2=scipy.integrate.quad(inttime,0.0,alim)
	if ans<(tautoday/10.**(7)):
		break

tauinit=ans
ainit=alim
xinit=np.log(ainit)
xfin=0.
numx=10000

#Set step vector for x for interpolation (slightly longer than for integration to ensure odeint doesn't crash)
xinterp=scipy.linspace(xinit,xfin+0.1,numx)
#Set step vector for x
xstep=scipy.linspace(xinit,xfin,numx)

######################## Calc eta in terms of x ######################################################################

def inteta(xi):
		return 1./cf.H1(xi,params)

etaans=np.zeros(numx)
for i in range(0,numx):
	res, err3=scipy.integrate.quad(inteta,-np.inf,xstep[i])
	etaans[i]=res

# CAROLA this is a slow integrator, recommend changing to scipy.integrate.simps or trapz instead if speed is an issue.
# Calculate a version of eta which applies slightly later than today, to be able to use odeint without problem.
etainterp=np.zeros(numx)
for i in range(0,numx):
	res, err3=scipy.integrate.quad(inteta,-np.inf,xinterp[i])
	etainterp[i]=res

# CAROLA probably better / faster to pass this information directly to where it is needed rather than saving it and reloading it elsewhere.
saveeverything=np.column_stack((xstep,xinterp,etaans,etainterp))
np.savetxt('./eta.txt',saveeverything)
