# This is a seperate file which produces the reionisation curve for the Callin code so we don't have to produce it every time we run

# Import libraries

import scipy.integrate
import scipy.interpolate
import pylab
import cosmofuncsoptdep as cf
import numpy as np
import math

#****************************************************************************************************************************#
#****************************************************************************************************************************#
#******************************************** DEFINITIONS AND SETUP *********************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#

# CAROLA, cosmological parameters need to be passed in at each step in chain and not fixed as here.
# Probably easier to call this directly at each step instead of saving result then loading elsewhere.

# Load cosmology from separate text file
alpha_C_tilde , HH0 , Nnu , wM0 , wB0 , A_s109 , n_s =np.loadtxt('./4DEGB_version/UniverseVars.txt',unpack=True)
# 4DEGB parameter alpha_C_tilde
# HH0, critical density, units km/(s*MpC)
#Number of species of massless neutrinos
# Dimensionless dark matter density/h^2
# Dimensionless baryonic matter density/h^2
# A_s109 = As*10^9

# Choice of paramterers: either
# alpha_C_tilde , HH0 , Nnu , wM0 , wB0 , A_s109 , n_s or
# with planck values __, 67.32, __, 0.12011, 0.022383, 2.101, 0.96605
# alpha_C_tilde ,zLambda , Nnu , zeq , wB0 , A_s109 , n_s
# with planck values __, 1.296678138994386, __, 3406.02315292002, 0.022383, 2.101, 0.96605

#Small h, Hubble constant
h=HH0/100.
wR0 = 2.4728905782994593*10**(-5)

####  FOR SECOND PARAMETER CHOICE: not sure it works ####
wN0GR = Nnu*(7./8.)*(4./11.)**(4./3.)*wR0	# neutrinos only
#alpha_C_tildeh2 = alpha_C_tilde*0.6732**2 # alpha_C_tilde*h**2 for a fiducial h from Planck
#wM0 = (alpha_C_tildeh2 + wR0+wN0GR)*(1+zeq) - wB0 #(wN0GR+wR0)/((1-alpha_C_tilde)/(zeq+1) - alpha_C_tilde*(zLambda+1)**3 - alpha_C_tilde) - wB0
#h = ((zLambda+1)**3 * (wB0 +wM0) + wB0 + wM0 + wR0 + wN0GR + alpha_C_tildeh2)**0.5 #(((zLambda + 1)**3 * (wB0 + wM0) + wB0 + wM0 + wR0 + wN0GR)/(1-alpha_C_tilde))**0.5
#HH0 = 100. * h
#########################################################


#Current values of energy densities
OmR0=wR0/(HH0/100.)**2
OmN0=Nnu*(7./8.)*(4./11.)**(4./3.)*OmR0 + alpha_C_tilde  # neutrinos + alpha_C_tilde combined here
OmM0=wM0/(HH0/100.)**2
OmB0=wB0/(HH0/100.)**2
OmL0=1.-OmM0-OmB0-OmN0-OmR0


# Constants used to get optical depth
Delta2s1s=8.227 # units s^(-1)
eps=13.605698 # units eV
xi0=24.5874 #units eV
xi1=54.42279 #units eV
me=9.10938291*10**(-31) # units kg
T0=2.725 # units K
alpha=1./137. #fine structure
pi=np.pi
G=6.67428*10**(-11) #units m^3kg^(-1)s^(-2)
mH=1.673575*10**(-27) # units kg
sigmat=6.6524616*10**(-29) #Thomson cross section, units m^2

# Conversion factors
kb=8.6173324*10**(-5) # units eV
hb= 1.054571726*10**(-34)#6.626068*10**(-34) units kg m^2 s^(-1) 
c=2.99792458*10**(8) # units m/s
evJ=1.6*10**(-19) #units J/eV
MpCm=3.085678*10**22 #units m/MpC

#Current value of the hubble constant in units of h/MpC, this sets our units.
H0=10**5/c #units h/MpC

# Stick everything in a vector
params=[Delta2s1s,eps,me,T0,alpha,kb,hb,c,evJ,MpCm,OmM0,OmR0,OmB0,OmL0, H0,OmN0]
	
# Critical density
rhocrit=3*(HH0*1000./MpCm)**2/(8.*pi*G) #units kg/m^3
print("rhocrit=",rhocrit*1000./100**3)

#Helium fraction
Yp=0.24

#Load data already produced in etacalc.py
xstep,xinterp,etaans,etainterp=np.loadtxt('./4DEGB_version/eta.txt',unpack=True)
numx=len(xstep)

#****************************************************************************************************************************#
#****************************************************************************************************************************#
#******************************************** OPTICAL DEPTH CALCULATOR ******************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#
	
################################################### Saha equation  #####################################################
	
# This is an appropriate approximation at very early times - > Use until Xe=0.99
	
#Preliminary definitions (we have factored our functions slightly differently than in Callin)
nBm=OmB0*rhocrit/(mH) # units m^(-3), not actually nB because we have factored out a^(-3)
Tb=T0/(np.exp(xinterp)) #units K

#Inital value of fe before recursion relation
fe=1.

#Construct the Saha Equation
factor=(me*Tb*kb*evJ*c**(-2)/(2.*pi))**(1.5)*(c/hb)**3 #units m^(-3)
A=4.*factor*np.exp(-xi1/(Tb*kb))
B=factor*np.exp(-eps/(Tb*kb))
C=2.*factor*np.exp(-xi0/(Tb*kb))

for i in range(0,10):
	
    #Dimensionless number density ratios
    x1=C/(fe*nBm*np.exp(-3*xinterp)+C+(A*C/(fe*nBm*np.exp(-3*xinterp)))) #He+ to He
    x2=A*C/(fe*nBm*np.exp(-3*xinterp)*(fe*nBm*np.exp(-3*xinterp)+C+(A*C/(fe*nBm*np.exp(-3*xinterp))))) #He++ to He
    xH=B/(fe*nBm*np.exp(-3*xinterp)+B)  #H+ to H
    fenew=(2.*x2+x1)*Yp*0.25+xH*(1-Yp)
    fenewsave=np.column_stack((xinterp,fenew/(1.-Yp)))
    np.savetxt('./4DEGB_version/fenew.txt',fenewsave)
    fe=fenew
	#It's fine that for fe[-1], fe will be nan, because we will never use it at x that high, but instead use the full ode solution

Xesa=fe/(1.-Yp)

#Find the first index where Xesa<0.99
indexsa=next(j[0] for j in enumerate(Xesa) if j[1]<0.99)
	
#Find x for which  this is true
switchx=xinterp[indexsa]
xpeeb=xinterp[indexsa:numx]

########################################### Peebles equation derivative #####################################################
	
def derivXe(z,x):
	
	#Define the proper name of our differentiation variable
	Xe=z
	
	#construct the derivative
	Tb=T0/(np.exp(x)) #units K
	phi2=0.448*np.log(eps/(Tb*kb)) #no units
	alpha2=64.*pi/(27.*pi)**(0.5)*alpha**2/(me**2)*(eps/(Tb*kb))**(0.5)*phi2 # kg^(-2)
	beta=alpha2*(me*Tb*kb*evJ*c**(-2)/(2*pi))**(1.5)*hb**(-1)*c**2 #s^(-1)
	beta2=beta*np.exp(-eps/(4.*Tb*kb)) #s^(-1)
	nHs=OmB0*rhocrit/(mH*np.exp(3.*x))*c**3 #s^(-3)
	nBs=OmB0*rhocrit/(mH*np.exp(3.*x))*c**3 #s^(-3)
	n1s=(1.-Xe)*(1-Yp)*nBs #s^(-3)
	Hnonconf=cf.H1(x,params)*h*c/(np.exp(x)*MpCm) #s^(-1)
	delalp=Hnonconf*(3.*eps*evJ/hb)**3/((8.*pi)**2*n1s) #s^(-1)
	Cr=(Delta2s1s+delalp)/(Delta2s1s+delalp+beta2) # no units
	diffXe=Cr/Hnonconf*(beta*np.exp(-eps/(Tb*kb))*(1-Xe)-(1-Yp)*nBs*alpha2*hb**2*c**(-4)*Xe**2) #no units
	
	return diffXe
	
######################################## Solve the Peebsle equation #######################################################
Xeinit=Xesa[indexsa]

Xesolpeebs=scipy.integrate.odeint(derivXe,Xeinit,xpeeb)

######################################## Format Solution #################################################################

#Attach Saha and Peebles
Xetot=np.append(Xesa[0:indexsa],Xesolpeebs)
redshift=np.exp(-xinterp)-1.
#Cut for plotting
indexstart=next(j[0] for j in enumerate(xinterp) if j[1]>(-17.5))
indexend=next(j[0] for j in enumerate(xinterp) if j[1]>(-0.001))
Xetotplot=Xetot[indexstart:indexend]
XeSahaplot=Xesa[indexstart:indexend]
redshiftplot=redshift[indexstart:indexend]
xplot=xinterp[indexstart:indexend]


########################### Calcualte optical depth from Xe and interpolate ###############################################

nBcalc=nBm*np.exp(-3.*xinterp)
ne=Xetot*nBcalc
taudot=-ne*(sigmat*MpCm)*np.exp(xinterp)/(cf.H1(xinterp,params)*h)
taudotplot=taudot[indexstart:indexend]
print( len(xplot))
print( len(taudotplot))

#Interpolate optical depth to be able to use given a different vector if we want
taudotfunc=scipy.interpolate.interp1d(xinterp,taudot)
nefunc=scipy.interpolate.interp1d(xinterp,ne)

savethese=np.column_stack((xstep,xinterp,Xetot,ne,taudot,etaans,etainterp))
np.savetxt('./4DEGB_version/Xevars.txt',savethese)

savethese2=np.column_stack((xplot,taudotplot))
np.savetxt('./4DEGB_version/XevarsHeplot.txt',savethese2)



