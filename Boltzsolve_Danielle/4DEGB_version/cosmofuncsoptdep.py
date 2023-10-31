""" Calculates various cosmological functions """

__author__="Danielle Leonard"
__date__="October 22, 2012"


import numpy as np

def H1(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, and Gamma as a tuple.
		
	Valid before recombination.
	
	Returns the current value of the conformal cosmological constant."""
	return params[14]*((params[10]+params[12])/np.exp(z)+(params[11]+params[15])/np.exp(z)**2+params[13]*np.exp(z)**2)**(0.5)

def OmR1(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, and Gamma as a tuple. 

	Valid before recombination
	
	Returns the current value of OmegaR, as in the radiation density."""
	return params[11]/((params[10]+params[12])*np.exp(z)+params[15]+params[11]+params[13]*np.exp(z)**4)

def OmM1(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, and Gamma as a tuple. 

	Valid before recombination.
	
	Returns the current value of OmegaM, as in the matter density.""" 
	return np.exp(z)*params[10]/((params[10]+params[12])*np.exp(z)+params[11]+params[15]+params[13]*np.exp(z)**4)

def OmB1(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, OmegaL0, OmegaB0 and Gamma as a tuple. 

	Valid before recombination.
	
	Returns the current value of OmegaB, as in the baryon density.""" 
	return np.exp(z)*params[12]/((params[10]+params[12])*np.exp(z)+params[11]+params[15]+params[13]*np.exp(z)**4)

def OmN1(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, and Gamma as a tuple. 

	Valid before recombination
	
	Returns the current value of OmegaN, as in the neutrino density."""
	return params[15]/((params[10]+params[12])*np.exp(z)+params[11]+params[15]+params[13]*np.exp(z)**4)
	
def OmN1GR(z,params):
	"""Takes the list z with 5 elements representing DeltaM, ThetaM, DeltaR, ThetaR, and a as defined in Bertschinger and Ma
	and the parameters OmegaM0,OmegaR0,H0, k, and Gamma as a tuple. 

	Valid before recombination
	
	Returns the current value of OmegaN, as in the neutrino density."""
	return params[16]/((params[10]+params[12])*np.exp(z)+params[11]+params[15]+params[13]*np.exp(z)**4)


