#This program attempts to calculate the matter power spectrum today using the method put forth in arxiv:astro-ph/0606683v1 by Petter Callin

# Import libraries

import scipy.integrate
import scipy.interpolate
import pylab
import cosmofuncsoptdep as cf
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


#****************************************************************************************************************************#
#****************************************************************************************************************************#
#******************************************** DEFINITIONS AND SETUP *********************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#

# CAROLA cosmological parameters of course need to be passed and changed at each step in the chain. 
# CHANGED define 4DEGB parameter alpha_C_tilde
alpha_C_tilde = 1e-5

# Constants used to get optical depth
Delta2s1s=8.227 # units s^(-1)
eps=13.605698 # units eV
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

# Critical density
HH0=68.14 #units km/(s*MpC)

rhocrit=3.*(HH0*1000./MpCm)**2/(8.*pi*G) #units kg/m^3 (March 5 agrees with Dodelson value page 6)

#Small h, Hubble constant
h=HH0/100.

#Number of species of massless neutrinos
Nnu=3.046

#Current values of energy densities

OmR0=2.47*10**(-5)/(h)**2
#CHANGED added GR version without alpha_C and 4DEGB version with alpha_C
OmN0=Nnu*(7./8.)*(4./11.)**(4./3.)*OmR0 + alpha_C_tilde  # neutrinos + alpha_C_tilde combined here
OmN0GR=Nnu*(7./8.)*(4./11.)**(4./3.)*OmR0   # neutrinos only
OmM0=0.11805/(HH0/100.)**2
OmB0=0.022242/(HH0/100.)**2
OmL0=1.-OmM0-OmB0-OmN0-OmR0

#Current value of the hubble constant in units of h/MpC, this sets our units.
H0=10**5/c #units h/MpC

# CHANGED - defined alpha_C
alpha_C = alpha_C_tilde * H0**2 # TO CHECK - for now units of (h/MpC)**2

# Stick everything in a vector
params=[Delta2s1s,eps,me,T0,alpha,kb,hb,c,evJ,MpCm,OmM0,OmR0,OmB0,OmL0, H0,OmN0,OmN0GR]


"""#specify our desired k range
klo=0.01
khi=0.1
ksp=0.05  #this is the LOGARITHMIC spacing in k

#calculate range of for loops and number of spots calcualted
rangea=int(-math.log10(khi)/ksp)
rangeb=int(-math.log10(klo)/ksp)
rangek=rangeb-rangea
numspots=rangek # Number of k evaluated

#Initialize array to hold power spectrum
PkDM=np.zeros(numspots)
kvec=np.zeros(numspots)"""

#Where to output data files?
dataloc='./4DEGB_version/'


#****************************************************************************************************************************#
#****************************************************************************************************************************#
#******************************************** OPTICAL DEPTH CALCULATOR ******************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#

# Load reionisation data
xstep,xinterp,Xetot,ne,taudot,etaans,etainterp=np.loadtxt('./4DEGB_version/Xevars.txt',unpack=True)
numx=len(xstep)

# Interpolate
taudotfunc=scipy.interpolate.interp1d(xinterp,taudot)
etafunc=scipy.interpolate.interp1d(xinterp,etainterp)

#Define variables:
xinit=xstep[0]
xfin=0.0

begrecomb=next(j[0] for j in enumerate(Xetot) if j[1]<0.99)

aval=np.exp(xstep)
OmB=aval*params[12]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
OmM=aval*params[10]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
OmR=params[11]/((params[10]+params[12])*aval+params[15]+params[11]+params[13]*aval**4)
OmN=params[15]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
#CHANGED added GR version without alpha_C
OmNGR=params[16]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
OmL=1.-OmB-OmM-OmR-OmN
Omtot=OmB+OmM+OmR+OmN
Omrad=OmR+OmN
Ommat=OmM+OmB
redshift=1./aval-1.


#Ombcol=np.column_stack((xstep,OmB,OmM,OmR,OmL,aval,etaans,Omtot,Omrad,Ommat,redshift))
#np.savetxt(dataloc+'OmB.txt',Ombcol)


#****************************************************************************************************************************#
#****************************************************************************************************************************#
#********************************************** SOLVE MAIN EQUATIONS  *******************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#

##################################### Calculate derivative of H #############################################################
Htry=cf.H1(xinterp,params)

#We can analytically calculate the derivative of Hder
Hder=H0*(-(OmM0+OmB0)*np.exp(-xstep)-2.*(OmR0+OmN0)*np.exp(-2.*xstep)+2.*OmL0*np.exp(2.*xstep))/((OmM0+OmB0)*np.exp(-xstep)+(OmR0+OmN0)*np.exp(-2.*xstep)+OmL0*np.exp(2.*xstep))**0.5

#Non interp
Hder2=H0*(-(OmM0+OmB0)*np.exp(-xinterp)-2.*(OmR0+OmN0)*np.exp(-2.*xinterp)+2.*OmL0*np.exp(2.*xinterp))/((OmM0+OmB0)*np.exp(-xinterp)+(OmR0+OmN0)*np.exp(-2.*xinterp)+OmL0*np.exp(2.*xinterp))**0.5

#Interpolate this
Hderfunc=scipy.interpolate.interp1d(xinterp,Hder)

##################################### Calculate derivative of Taudot #############################################################

#Use appoximation given in Callin
taudotder=-taudotfunc(xinterp)

#Interpolate this
taudotderfunc=scipy.interpolate.interp1d(xinterp,taudotder)


########################################## Tightly coupled DE solver #########################################################

def derivtightlycoupled(z,x,params):

	
	#Get x value for this call
	xpoint=x

	#Give things real names
	Delm=z[0]
	vm=z[1]
	Delb=z[2]
	vb=z[3]
	T0=z[4]
	theta1=z[5]
	N0=z[6]
	n1=z[7]
	n2=z[8]
	n3=z[9]
	n4=z[10]
	n5=z[11]
	n6=z[12]
	n7=z[13]
	n8=z[14]
	n9=z[15]
	n10=z[16]
	# CHANGED added the following two lines of code
	dphiprime = z[17] # derivative of 4DEGB scalar field wrt x=ln(a)
	dphi = z[18] # new scalar field in 4DEGB

	#Define things necessary for calculation
	aval=np.exp(xpoint)
	taudotdot=taudotderfunc(xpoint)
	tdot=taudotfunc(xpoint)
	Hdot=Hderfunc(xpoint)
	R=4.*OmR0/(3.*OmB0*aval)

	

	#Get values from cosmofuncs
	OmM=aval*params[10]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	OmR=params[11]/((params[10]+params[12])*aval+params[15]+params[11]+params[13]*aval**4)
	OmB=aval*params[12]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	OmN=params[15]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	#CHANGED added GR version without alpha_C
	OmNGR=params[16]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	Hconf=params[14]*((params[10]+params[12])/aval+(params[11]+params[15])/aval**2+params[13]*aval**2)**(0.5)

	kovH=k/Hconf
	
	# Photon multipoles which we will use in the de solver (except theta1)
	T0prime=-kovH*theta1
	theta2=-8.*kovH/(15.*tdot)*theta1
	
	# Get values for psi and the derivative of phi
	# CHANGED phiv=-1.5*Hconf**2*(OmM*(Delm-3.*Hconf*vm/k)+4.*OmR*(T0+3.*Hconf*theta1/k)+OmB*(Delb-3.*Hconf*vb/k)+4*OmN*(N0+3*Hconf*n1/k))/(k**2+4.5*Hconf**2*OmM+6.*Hconf**2*OmR+4.5*Hconf**2*OmB+6*Hconf**2*OmN)
	phiv= (-72.*(H0**2)*alpha_C*(OmR*theta2 + OmNGR*n2)/((k**2)*aval**4) -6.*alpha_C*Hconf*dphiprime/aval**2 -1.5*(Hconf**2 - alpha_C/aval**2)*(OmM*(Delm-3.*Hconf*vm/k)+4.*OmR*(T0+3.*Hconf*theta1/k)+OmB*(Delb-3.*Hconf*vb/k)+4*OmNGR*(N0+3*Hconf*n1/k))) / (k**2 + (Hconf**2 - alpha_C/aval**2)*(4.5*OmM+6.*OmR+4.5*OmB+6*OmNGR) - 6.*alpha_C/aval**2)
	
	#CHANGED added GR component only
	psiv=phiv-12.*H0**2/k**2/aval**2*(OmR0*theta2+OmN0GR*n2)
	
	# CHANGED phiprimev=-psiv-kovH**2/(3.)*phiv-H0**2/(2.*Hconf**2)*(OmM0/aval*(Delm+3.*phiv)+OmB0/aval*(Delb+3.*phiv)+4.*OmR0/aval**2*(T0+phiv)+4.*OmN0/aval**2*(N0+phiv))
	phiprimev= - psiv - kovH**2/(3.)*phiv + 2*alpha_C*psiv/(aval*Hconf)**2 - 2*alpha_C*dphiprime/(aval**2*Hconf) -H0**2/(2.*Hconf**2)*(OmM0/aval*(Delm+3.*phiv)+OmB0/aval*(Delb+3.*phiv)+4.*OmR0/aval**2*(T0+phiv)+4.*OmN0GR/aval**2*(N0+phiv))
	
	# Matter variables except vb
	Delmprime=kovH*vm
	vmprime=-vm-kovH*psiv
	Delbprime=kovH*vb

	#Derivatives of neutrino multipoles
	N0prime=-kovH*n1
	n1prime=kovH*((N0+phiv)/3.-2.*n2/3.+psiv/3.)
	n2prime=kovH*(2.*n1/(5.)-3.*n3/(5.))
	n3prime=kovH*(3.*n2/(7.)-4.*n4/(7.))
	n4prime=kovH*(4.*n3/(9.)-5.*n5/(9.))
	n5prime=kovH*(5.*n4/(11.)-6*n6/(11.))
	n6prime=kovH*(6.*n5/(13.)-7.*n7/(13.))
	n7prime=kovH*(7.*n6/(15.)-8.*n8/(15.))
	n8prime=kovH*(8.*n7/(17.)-9.*n9/(17.))
	n9prime=kovH*(9.*n8/(19.)-10.*n10/(19.))
	n10prime=kovH*n9-11./(Hconf*etafunc(xpoint))*n10
	
	theta2prime=0.0
	
	# CHANGED added 2 equations below for additional field in 4DEGB
	psiprimev=phiprimev-12.*H0**2/k**2/aval**2*(OmR0*(theta2prime - 2.*theta2)+OmN0GR*(n2prime - 2.*n2))
	dphiprimeprime = -Hdot*dphiprime/Hconf + (phiprimev + psiprimev)/Hconf - kovH**2 * dphi/3.

	#Use an analytically reduced version of equation 33 which we have manipulated to reduce errors TODO--does this change in 4DEGB?
	t1vbcomb=((2.*R*tdot)/((1.+R)*tdot+Hdot/Hconf-1.))*(3.*theta1+vb)-kovH/((1.+R)*tdot+Hdot/Hconf-1.)*psiv+(1.-Hdot/Hconf)*kovH/((1.+R)*tdot+Hdot/Hconf-1.)*(-(T0+phiv)+2.*theta2)+kovH/((1.+R)*tdot+Hdot/Hconf-1.)*(-T0prime-phiprimev+2.*theta2prime)
	vbprime=(-vb - kovH*psiv + R*(t1vbcomb + kovH*(-(T0+phiv)+2.*theta2) - kovH*psiv))/(1.+R)
	theta1prime=1./3.*(t1vbcomb-vbprime)
	
	#Return
	# CHANGED to include additional 4DEGB scalar field degrees of freedom
	derivatives=[Delmprime,vmprime,Delbprime,vbprime,T0prime,theta1prime,N0prime,n1prime,n2prime,n3prime,n4prime,n5prime,n6prime,n7prime,n8prime,n9prime,n10prime,dphiprimeprime,dphiprime]

	return derivatives

######################################### Regular DE solver ##################################################################

def deriv(z,x,params):

	xpoint=x

	#Name solution array
	Delm=z[0]
	vm=z[1]
	Delb=z[2]
	vb=z[3]
	T0=z[4]
	theta1=z[5]
	N0=z[6]
	n1=z[7]
	theta2=z[8]
	theta3=z[9]
	theta4=z[10]
	theta5=z[11]
	theta6=z[12]
	theta0p=z[13]
	theta1p=z[14]
	theta2p=z[15]
	theta3p=z[16]
	theta4p=z[17]
	theta5p=z[18]
	theta6p=z[19]
	n2=z[20]
	n3=z[21]
	n4=z[22]
	n5=z[23]
	n6=z[24]
	n7=z[25]
	n8=z[26]
	n9=z[27]
	n10=z[28]
	# CHANGED added the following two lines of code
	dphiprime = z[29] # derivative of 4DEGB scalar field wrt x=ln(a)
	dphi = z[30] # new scalar field in 4DEGB

	#Get values of certain things to speed up evaluations
	aval=np.exp(xpoint)
	Hdot=Hderfunc(xpoint)

	#Get values from cosmofuncs
	OmM=aval*params[10]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	OmR=params[11]/((params[10]+params[12])*aval+params[15]+params[11]+params[13]*aval**4)
	OmB=aval*params[12]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	OmN=params[15]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	#CHANGED added GR version without alpha_C
	OmNGR=params[16]/((params[10]+params[12])*aval+params[11]+params[15]+params[13]*aval**4)
	Hconf=params[14]*((params[10]+params[12])/aval+(params[11]+params[15])/aval**2+params[13]*aval**2)**(0.5)

	#Get phi and psi values
	# CHANGED phiv=-1.5*Hconf**2*(OmM*(Delm-3.*Hconf*vm/k)+4.*OmR*(T0+3.*Hconf*theta1/k)+OmB*(Delb-3.*Hconf*vb/k)+4*OmN*(N0+3*Hconf*n1/k))/(k**2+4.5*Hconf**2*OmM+6.*Hconf**2*OmR+4.5*Hconf**2*OmB+6.*Hconf**2*OmN)
	phiv=(-72.*(H0**2)*alpha_C*(OmR*theta2 + OmNGR*n2)/((k**2)*aval**4) -6.*alpha_C*Hconf*dphiprime/aval**2 -1.5*(Hconf**2 - alpha_C/aval**2)*(OmM*(Delm-3.*Hconf*vm/k)+4.*OmR*(T0+3.*Hconf*theta1/k)+OmB*(Delb-3.*Hconf*vb/k)+4*OmNGR*(N0+3*Hconf*n1/k))) / (k**2 + (Hconf**2 - alpha_C/aval**2)*(4.5*OmM+6.*OmR+4.5*OmB+6*OmNGR) - 6.*alpha_C/aval**2)
	
	#CHANGED added GR component only
	psiv=phiv-12.*H0**2/k**2/aval**2*(OmR0*theta2+OmN0GR*n2)
	
	# CHANGED phiprimev=-psiv-k**2/(3.*Hconf**2)*phiv-H0**2/(2.*Hconf**2)*(OmM0/aval*(Delm+3.*phiv)+OmB0/aval*(Delb+3.*phiv)+4.*OmR0/aval**2*(T0+phiv)+4*OmN0/aval**2*(N0+phiv))
	kovH = k/Hconf
	phiprimev= - psiv - kovH**2/(3.)*phiv + 2*alpha_C*psiv/(aval*Hconf)**2 - 2*alpha_C*dphiprime/(aval**2*Hconf) -H0**2/(2.*Hconf**2)*(OmM0/aval*(Delm+3.*phiv)+OmB0/aval*(Delb+3.*phiv)+4.*OmR0/aval**2*(T0+phiv)+4.*OmN0GR/aval**2*(N0+phiv))

	#Define auxiliary values
	R=4.*OmR0/(3.*OmB0*aval)
	BPi=theta2+theta2p+theta0p
	taudota=taudotfunc(xpoint)

	#Define a commonly used expression:
	kovH=k/Hconf

	#Derivatives
	T0prime=-kovH*theta1
	theta1prime=kovH*(T0+phiv)/3. - 2.*kovH*theta2/3. + kovH*psiv/(3.) + taudota*(theta1+vb/3.)
	theta2prime=2.*kovH/5.*theta1 - 3.*kovH/5.*theta3 + taudota*(theta2-0.1*BPi)
	theta3prime=3.*kovH/7.*theta2 - 4.*kovH/7.*theta4 + taudota*theta3
	theta4prime=4.*kovH/9.*theta3 - 5.*kovH/9.*theta5 + taudota*theta4
	theta5prime=5.*kovH/11.*theta4 - 6.*kovH/11.*theta6 + taudota*theta5
	theta6prime=kovH*theta5 - 7.*theta6/Hconf/etafunc(xpoint) + taudota*theta6
	Delmprime=kovH*vm
	vmprime= -vm - kovH*psiv
	Delbprime= kovH*vb
	vbprime= -vb - kovH*psiv + taudota*R*(3.*theta1+vb)
	theta0pprime= -kovH*theta1p + taudota*(theta0p-0.5*BPi)
	theta1pprime= kovH*theta0p/(3.) - 2.*kovH*theta2p/(3.) + taudota*theta1p
	theta2pprime= 2.*kovH*theta1p/(5.) - 3.*kovH*theta3p/(5.) + taudota*(theta2p-0.1*BPi)
	theta3pprime= 3.*kovH*theta2p/(7.) - 4.*kovH*theta4p/(7.) + taudota*theta3p
	theta4pprime= 4.*kovH*theta3p/(9.) - 5.*kovH*theta5p/(9.) + taudota*theta4p
	theta5pprime= 5.*kovH*theta4p/(11.) - 6.*kovH*theta6p/(11.) + taudota*theta5p
	theta6pprime= kovH*theta5p - 7./(Hconf*etafunc(xpoint))*theta6p + taudota*theta6p
	N0prime=-kovH*n1
	n1prime=kovH*((N0+phiv)/3.-2./(3.)*n2+psiv/3.)
	n2prime=kovH*(2.*n1/5.-3.*n3/(5.))
	n3prime=kovH*(3.*n2/7.-4.*n4/(7.))
	n4prime=kovH*(4.*n3/(9.)-5.*n5/(9.))
	n5prime=kovH*(5.*n4/(11.)-6*n6/(11.))
	n6prime=kovH*(6.*n5/(13.)-7.*n7/(13.))
	n7prime=kovH*(7.*n6/(15.)-8.*n8/(15.))
	n8prime=kovH*(8.*n7/(17.)-9.*n9/(17.))
	n9prime=kovH*(9.*n8/(19.)-10.*n10/(19.))
	n10prime=kovH*n9-11./(Hconf*etafunc(xpoint))*n10
	
	# CHANGED added 2 equations below for additional field in 4DEGB
	psiprimev=phiprimev-12.*H0**2/k**2/aval**2*(OmR0*(theta2prime - 2.*theta2)+OmN0GR*(n2prime - 2.*n2))
	dphiprimeprime = -Hdot*dphiprime/Hconf + (phiprimev + psiprimev)/Hconf - kovH**2 * dphi/3.
	
	# CHANGED to include additional 4DEGB scalar field degrees of freedom
	#Return
	derivatives=[Delmprime,vmprime,Delbprime,vbprime,T0prime,theta1prime,N0prime,n1prime,theta2prime,theta3prime,theta4prime,theta5prime,theta6prime,theta0pprime,theta1pprime,theta2pprime,theta3pprime,theta4pprime, theta5pprime,theta6pprime,n2prime,n3prime,n4prime,n5prime,n6prime,n7prime,n8prime,n9prime,n10prime,dphiprimeprime,dphiprime]
	return derivatives


###################################### Loops starts here #####################################################################

#Read in camb file to use their k values or set k directly.
kload2 = np.logspace(-4,1,100)


transfermattertoday=np.zeros(len(kload2))
matterpowertestnewtoday=np.zeros(len(kload2))
transfercdm0=np.zeros(len(kload2))
transferbar0=np.zeros(len(kload2))
kvec=np.zeros(len(kload2))
phiofk=np.zeros(len(kload2))

indexk=0
for k in tqdm(kload2):
    print( "k=", k)



####################### Check where the tight coupling approximation should be used. #########################################
	
    #Define the two conditions (other than recombination)
    check1=np.absolute(k/(cf.H1(xstep,params)*taudot))
    check2=np.absolute(taudot)

	#Get the indices where both conditions fail
    check1bad=next(j[0] for j in enumerate(check1) if j[1]>0.1)
    check2bad=next(j[0] for j in enumerate(check2) if j[1]<10.)

	#Pick which one occurs first
    if check1bad<check2bad:
        tightcoupend=check1bad-1
    else:
        tightcoupend=check2bad-1

    #If this is after the start of recombination, use recombination as the end of tight coupling, where we define recombination as beginning at the point where Xe=0.95
    #if Xetot[tightcoupend]<0.95:
    #	tightcoupend=begrecomb

    #Break up the array for solving the DE's into tight coupling and regular
    xtight=xstep[0:tightcoupend]
    xreg=xstep[tightcoupend:10000]

###################################################### Initial conditions, tight coupling #############################################

    fv=1./((8./(7.*Nnu))*(11./4.)**(4./3.)+1.)
    Hconf=cf.H1(xstep,params)

    #Initial conditions with C=1/2, chi (camb)=1
    # CAROLA, these change for 4DEGB
    
    # CHANGED added this for the additional scalar field
    ddphi_detai = 1.    ### NOTE: FOR NOW ASSUME D(DPHI)/DETA = CONST
    dphidoti = ddphi_detai/cf.H1(xinit,params)  ### Assume const ddphi_detai
    dphii = dphidoti     ### Assume const ddphi_detai
    
    Cinit=-0.5
    psii=20.*Cinit/(15.+4.*fv)
    phii=(1.+0.4*fv)*psii - 4*fv*alpha_C_tilde/(5*(OmR0+OmN0)) * ddphi_detai
    

    # CHANGED theta0i=-0.5*psii
    theta0i = -0.5*((OmR0+OmN0GR-alpha_C_tilde)*psii/(OmR0+OmN0GR) + 2.*alpha_C_tilde/(OmR0+OmN0GR) *ddphi_detai)
    T0i=theta0i-psii
    # CHANGED delmi=-1.5*psii
    delmi = 3*theta0i
    Delmi=delmi-3*psii
    delbi=delmi
    Delbi=Delmi
    # CHAGNED theta1i=k/6./cf.H1(xinit,params)*psii
    theta1i=k/6./cf.H1(xinit,params)*psii * ((OmR0+OmN0)/(2*(OmR0+OmN0GR)) - 2.*alpha_C_tilde/(OmR0+OmN0GR) *ddphi_detai/psii)
    # CHANGED vmi=-k/2./cf.H1(xinit,params)*psii
    vmi= -3*theta1i
    vbi=vmi
    #CHANGED N0i=-1.5*psii
    N0i= T0i
    n1i=theta1i
    n2i=k**2*np.exp(2.*xinit)/((12.*H0**2*OmN0)*((5./(2.*fv))+1.)) *(phii + 2*alpha_C_tilde*ddphi_detai/(OmR0+OmN0))
    n3i=k/(7.*cf.H1(xinit,params))*n2i
    n4i=k/(9.*cf.H1(xinit,params))*n3i
    n5i=k/(11.*cf.H1(xinit,params))*n4i
    n6i=k/(13.*cf.H1(xinit,params))*n5i
    n7i=k/(15.*cf.H1(xinit,params))*n6i
    n8i=k/(17.*cf.H1(xinit,params))*n7i
    n9i=k/(19.*cf.H1(xinit,params))*n8i
    n10i=k/(21.*cf.H1(xinit,params))*n9i
    

############################################ Solve the ODE, tight coupling #############################################################
    #print 'begin tight couple'
    # CHANGED to include additional 4DEGB scalar field degrees of freedom
    inittight=[Delmi,vmi,Delbi,vbi,T0i,theta1i,N0i,n1i,n2i,n3i,n4i,n5i,n6i,n7i,n8i,n9i,n10i,dphidoti,dphii]
    soltight=scipy.integrate.odeint(derivtightlycoupled,inittight,xtight,args=(params,))
    #print 'end tight couple'

####################################### Initial conditions for regular solver ##########################################################

    #Construct higher photon multipoles and all photon polarisation multipoles from tight coupling
    theta2tight=-8.*k/(15.*cf.H1(xtight,params)*taudotfunc(xtight))*soltight[:,5]
    theta3tight=-3.*k*theta2tight/(7.*cf.H1(xtight,params)*taudotfunc(xtight))
    theta4tight=-4.*k*theta3tight/(9.*cf.H1(xtight,params)*taudotfunc(xtight))
    theta5tight=-5.*k*theta4tight/(11.*cf.H1(xtight,params)*taudotfunc(xtight))
    theta6tight=-6.*k*theta5tight/(13.*cf.H1(xtight,params)*taudotfunc(xtight))

    theta0ptight=5./4.*theta2tight
    theta1ptight=-k/(4.*cf.H1(xtight,params)*taudotfunc(xtight))*theta2tight
    theta2ptight=0.25*theta2tight
    theta3ptight=-3.*k/(7.*cf.H1(xtight,params)*taudotfunc(xtight))*theta2ptight
    theta4ptight=-4.*k/(9.*cf.H1(xtight,params)*taudotfunc(xtight))*theta3ptight
    theta5ptight=-5*k/(11.*cf.H1(xtight,params)*taudotfunc(xtight))*theta4ptight
    theta6ptight=-6*k/(13.*cf.H1(xtight,params)*taudotfunc(xtight))*theta5ptight


    #Initial conditions in non-tight coupling regime
    T0i2=soltight[:,4][-1]
    Delmi2=soltight[:,0][-1]
    Delbi2=soltight[:,2][-1]
    theta1i2=soltight[:,5][-1]
    vmi2=soltight[:,1][-1]
    vbi2=soltight[:,3][-1]
    theta2i2=theta2tight[-1]
    theta3i2=theta3tight[-1]
    theta4i2=theta4tight[-1]
    theta5i2=theta5tight[-1]
    theta6i2=theta6tight[-1]
    theta0pi2=theta0ptight[-1]
    theta1pi2=theta1ptight[-1]
    theta2pi2=theta2ptight[-1]
    theta3pi2=theta3ptight[-1]
    theta4pi2=theta4ptight[-1]
    theta5pi2=theta5ptight[-1]
    theta6pi2=theta6ptight[-1]
    N0i2=soltight[:,6][-1]
    n1i2=soltight[:,7][-1]
    n2i2=soltight[:,8][-1]
    n3i2=soltight[:,9][-1]
    n4i2=soltight[:,10][-1]
    n5i2=soltight[:,11][-1]
    n6i2=soltight[:,12][-1]
    n7i2=soltight[:,13][-1]
    n8i2=soltight[:,14][-1]
    n9i2=soltight[:,15][-1]
    n10i2=soltight[:,16][-1]
    
    # CHANGED to include additional 4DEGB scalar field
    dphidoti2=soltight[:,17][-1]
    dphii2=soltight[:,18][-1]


############################################ Solve the ODE, regular #############################################################

    initreg=[Delmi2,vmi2,Delbi2,vbi2,T0i2,theta1i2,N0i2,n1i2,theta2i2,theta3i2,theta4i2,theta5i2,theta6i2,theta0pi2,theta1pi2,theta2pi2,theta3pi2,theta4pi2,theta5pi2,theta6pi2,n2i2,n3i2,n4i2,n5i2,n6i2,n7i2,n8i2,n9i2,n10i2,dphidoti2,dphii2]
    #print 'begin solve reg ode'
    solreg=scipy.integrate.odeint(deriv,initreg,xreg,args=(params,))
    #print 'end solve reg ode'


#****************************************************************************************************************************#
#****************************************************************************************************************************#
#***************************************************** OUTPUT  **************************************************************#
#****************************************************************************************************************************#
#****************************************************************************************************************************#

############################################ Patch regimes together ##############################################################


    #Patch together
    Delmtot=np.append(soltight[:,0],solreg[:,0])
    vmtot=np.append(soltight[:,1],solreg[:,1])
    Delbtot=np.append(soltight[:,2],solreg[:,2])
    vbtot=np.append(soltight[:,3],solreg[:,3])
    T0tot=np.append(soltight[:,4],solreg[:,4])
    theta1tot=np.append(soltight[:,5],solreg[:,5])
    theta2tot=np.append(theta2tight,solreg[:,8])
    theta3tot=np.append(theta3tight,solreg[:,9])
    theta4tot=np.append(theta4tight,solreg[:,10])
    theta5tot=np.append(theta5tight,solreg[:,11])
    theta6tot=np.append(theta6tight,solreg[:,12])
    theta0ptot=np.append(theta0ptight,solreg[:,13])
    theta1ptot=np.append(theta1ptight,solreg[:,14])
    theta2ptot=np.append(theta2ptight,solreg[:,15])
    theta3ptot=np.append(theta3ptight,solreg[:,16])
    theta4ptot=np.append(theta4ptight,solreg[:,17])
    theta5ptot=np.append(theta5ptight,solreg[:,18])
    theta6ptot=np.append(theta6ptight,solreg[:,19])
    N0tot=np.append(soltight[:,6],solreg[:,6])
    n1tot=np.append(soltight[:,7],solreg[:,7])
    n2tot=np.append(soltight[:,8],solreg[:,20])
    n3tot=np.append(soltight[:,9],solreg[:,21])
    n4tot=np.append(soltight[:,10],solreg[:,22])
    n5tot=np.append(soltight[:,11],solreg[:,23])
    n6tot=np.append(soltight[:,12],solreg[:,24])
    n7tot=np.append(soltight[:,13],solreg[:,25])
    n8tot=np.append(soltight[:,14],solreg[:,26])
    n9tot=np.append(soltight[:,15],solreg[:,27])
    n10tot=np.append(soltight[:,16],solreg[:,28])
    
    # CHANGED to include additional 4DEGB scalar field
    dphidottot=np.append(soltight[:,17],solreg[:,29])
    dphitot=np.append(soltight[:,18],solreg[:,30])

    #Reconstruct phi

    OmM=cf.OmM1(xstep,params)
    OmR=cf.OmR1(xstep,params)
    OmB=cf.OmB1(xstep,params)
    OmN=cf.OmN1(xstep,params)
    OmNGR = cf.OmN1GR(xstep,params)
    Hconf=cf.H1(xstep,params)

    # CHANGED phicon=-1.5*Hconf**2*(OmM*(Delmtot-3.*Hconf*vmtot/k)+4.*OmR*(T0tot+3.*Hconf*theta1tot/k)+OmB*(Delbtot-3.*Hconf*vbtot/k)+4.*OmN*(N0tot+3.*Hconf*n1tot/k))/(k**2+4.5*Hconf**2*OmM+6.*Hconf**2*OmR+4.5*Hconf**2*OmB+6.*Hconf**2*OmN)
    
    phicon=(-72.*(H0**2)*alpha_C*(OmR*theta2tot + OmNGR*n2tot)/((k**2)*np.exp(4*xstep)) -6.*alpha_C*Hconf*dphidottot/np.exp(2*xstep) -1.5*(Hconf**2 - alpha_C/np.exp(2*xstep))*(OmM*(Delmtot-3.*Hconf*vmtot/k)+4.*OmR*(T0tot+3.*Hconf*theta1tot/k)+OmB*(Delbtot-3.*Hconf*vbtot/k)+4*OmNGR*(N0tot+3*Hconf*n1tot/k))) / (k**2 + (Hconf**2 - alpha_C/np.exp(2*xstep))*(4.5*OmM+6.*OmR+4.5*OmB+6*OmNGR) - 6.*alpha_C/np.exp(2*xstep))
    
    #CHANGED added GR component only
    psicon=phicon-12*H0**2/k**2/np.exp(2*xstep)*(OmR0*theta2tot+OmN0GR*n2tot)
	
    delmtot=Delmtot+3*phicon
    delbtot=Delbtot+3*phicon
    theta0=T0tot+phicon
    n0tot=N0tot+phicon

################################################## Save Power Spectrum Values #######################################################
    delMpowspect=OmM/(OmM+OmB)*delmtot+OmB/(OmM+OmB)*delbtot
    delMpowspectinv=OmM/(OmM+OmB)*(delmtot-3*Hconf*vmtot/k)+OmB/(OmM+OmB)*(delbtot-3*Hconf*vbtot/k)
    delMpowspectinvotherdef=OmM/(OmM+OmB)*(delmtot-3*Hconf*vmtot/k)**2+OmB/(OmM+OmB)*(delbtot-3*Hconf*vbtot/k)**2
    transfermatter=((delmtot-3*Hconf*vmtot/k)*OmM+(delbtot-3*Hconf*vbtot/k)*OmB)/(OmM+OmB)/k**2
    transfercdm=(delmtot-3*Hconf*vmtot/k)/k**2
    transferbar=(delbtot-3*Hconf*vbtot/k)/k**2
    matterpowertestnew=(((delmtot-3*Hconf*vmtot/k)*OmM+(delbtot-3*Hconf*vbtot/k)*OmB)/(OmM+OmB))**2*2.*np.pi**2*h**3/k**3*(2.1*10.**(-9)*np.exp((0.96-1.)*np.log(k/0.05)))

    transfermattertoday[indexk]=transfermatter[-1]
    transfercdm0[indexk]=transfercdm[-1]
    transferbar0[indexk]=transferbar[-1]
    matterpowertestnewtoday[indexk]=matterpowertestnew[-1]
    kvec[indexk]=k
    phiofk[indexk]=phicon[-1]

    indexk+=1
	

################################################## Output to file ###################################################################


matterpowertestnewsave=np.column_stack((kvec,matterpowertestnewtoday))
np.savetxt('./4DEGB_version/matterpower.txt',matterpowertestnewsave)

transfersave=np.column_stack((kvec,transfermattertoday))
np.savetxt('./4DEGB_version/transfermatter.txt',transfersave)

transfercdmsave=np.column_stack((kvec,transfercdm0))
np.savetxt('./4DEGB_version/transfercdm.txt',transfercdmsave)

transferbarsave=np.column_stack((kvec,transferbar0))
np.savetxt('./4DEGB_version/transferbar.txt',transferbarsave)



