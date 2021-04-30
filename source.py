# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

# modified from phienics - https://github.com/scaramouche-00/phienics

from dolfin import Expression, assemble, dx

import dolfin as d

import mpmath as mp
import numpy as np


class Source(object):

    def __init__( self, fem, physics ):

        self.fem = fem
        self.physics = physics
        
        # source radius, physical and rescaled units
        self.Rs = self.physics.Rs
        self.rs = 1.

        # mean density and mass, in physical units
        self.Rho_bar = self.physics.Rho_bar
        self.Ms = self.physics.Ms

        # rescaled mean density
        self.vd = self.physics.g * self.rs**self.physics.D  # volume of D-sphere in rescaled units 
        self.rho_bar = self.Rho_bar * self.physics.Vd / self.vd

        self.rho = None







class StepSource(Source):

    def __init__( self, fem, physics ):
        
        
        Source.__init__( self, fem, physics )
        
        self.rho = None
        self.build_source()


    def build_source( self ):

        self.rho = Expression('x[0] < rs ? rho_bar : 0.',
                                  degree=self.fem.func_degree, rho_bar=self.rho_bar, rs=self.rs )






        



class TopHatSource(Source):


    def __init__( self, fem, physics, w=0.02 ):

        Source.__init__( self, fem, physics )

        self.w = w
        self.A = None 

        self.build_source()


        

    def polylog_term( self ):
        # the Fermi-Dirac integral used in the normalisation of the source profile

        mu = self.rs/self.w
        
        # For very steep profiles/large mu, the argument of polylog becomes large and overflows:
        # use asymptotic series in this case.
        # The exact and asymptotic-series calculations match within machine precision for mu >~ 30
        if mu < 30.:
            return mp.fp.polylog( 3, -np.exp(mu) )
        else:
            return -1./6. * ( mu**3 + np.pi**2 * mu )


    def build_source( self ):

        # set normalisation
        self.A = self.Ms / ( -8. * np.pi * self.w**3 * self.polylog_term() )
        
        self.rho = Expression(' A / ( exp( (x[0] - t * rs)/w ) + 1. )',
                              degree=self.fem.func_degree, A=self.A,
                              rs=self.rs, w=self.w )






class CosSource(Source):

    

    def __init__( self, fem, physics ):
        
        Source.__init__( self, fem, physics )

        self.A = None # set in build_source
        
        self.rho = None
        self.build_source()


    def build_source( self ):

        rho =  Expression( 'x[0] < rs ? cos( pi * x[0]/rs ) + 1. : 0.',
                                     domain=self.fem.S, degree=self.fem.func_degree, rs=self.rs )
        r = Expression('x[0]', degree=self.fem.func_degree )
        N = self.sd * d.assemble( rho * r**(self.physics.D-1) * dx )
        self.A = self.Ms / N
        
        self.rho = Expression( 'x[0] < rs ? A * ( cos( pi * x[0]/rs ) + 1. ) : 0.',
                                     degree=self.fem.func_degree, A=self.A, rs=self.rs )









        


class GaussianSource(Source):

    def __init__( self, fem, physics, sigma=None, k=None ):

        Source.__init__( self, fem, physics, Rho_bar, Ms, Rs )

        self.sigma = sigma
        self.k = k
        self.check_params()

        self.rho = None
        self.build_source()


    def check_params( self ):

        if self.sigma is not None:
            if self.k is not None:
                message = "The parameters sigma and k cannot be simultaneously set: please set either of them."
                raise ValueError, message
            else:
                self.k = self.rs / self.sigma
        else:
            if self.k is None:
                message = "At least one parameter between sigma and k must be set: please set either of them."
                raise ValueError, message
            else:
                self.sigma = self.rs / self.k


    def build_source( self ):

        self.rho = Expression('Ms / pow( sqrt(2*pi) * sigma, D ) * exp( -0.5 * pow(x[0]/sigma,2) )',
                              Ms=self.Ms, D=self.physics.D, sigma=self.sigma, degree=self.fem.func_degree )

        

            

            
