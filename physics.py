# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

from scipy.special import gamma
import numpy as np


class SymPhysics(object):

    def __init__( self, D, coupling, mu, lam, M, Rs, Ms=None, Rho_bar=None ):

        # dimensions
        self.D = D

        # field params

        # type of coupling
        self.coupling = coupling
        if self.coupling not in ['linear','quadratic']:
            message = "Wrong choice of coupling. Please choose between 'linear' and 'quadratic'."
            raise ValueError, message

        # mass scales and nonlinear param
        self.mu = mu
        self.M = M
        self.lam = lam
  
        # geometrical quantities
        
        # volume geometric factor - D sphere
        self.g = np.pi**(self.D/2.) / gamma( self.D/2. + 1. )
        # surface geometric factor - (D-1) sphere
        self.sd = 2. * np.pi**(self.D/2.) / gamma( self.D/2. )

        # source radius
        self.Rs = Rs   
        # volume of D-sphere of source radius  
        self.Vd = self.g * self.Rs**self.D 

        # either mass or mean density of the source
        self.Ms = Ms
        self.Rho_bar = Rho_bar

        if self.Ms is None:
            self.Ms = self.Rho_bar * self.Vd

        elif self.Rho_bar is None:
            self.Rho_bar = self.Ms / self.Vd
            
        else:
            message = "Not enough information on the source mass: please set either Rho_bar or Ms."
            raise ValueError, message

        
        # source mass in natural units    
        if self.coupling=='linear':
            self.ms = self.Ms * self.mu**(self.D-3) * np.sqrt(self.lam) / self.M
        elif self.coupling=='quadratic':
            self.ms =self.Ms * self.mu**(self.D-2) / self.M**2
            

        # useful derived quantities
        # vev - physical units
        self.Vev = self.mu / np.sqrt( self.lam )
        # source radius in units of Compton wavelength
        self.x0 = self.mu * self.Rs

