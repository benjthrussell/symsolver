# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

from initial_guess import InitialGuess

from scipy.special import k0, k1, i0, i1, lambertw

import dolfin as d
import numpy as np



class OldAnalytics(InitialGuess):

    def __init__( self, fem, source, physics,
                  mn=None, mf=None, phi_at_infty='zero' ):

        user_set_part = 2

        InitialGuess.__init__( self, fem, source, physics,
                               mn, mf, phi_at_infty, user_set_part )

        self.old_charge = None
        self.compute_old_charge()




    def compute_old_charge( self ):

        phi_m, rho_eff, x0 = self.phi_m, self.rho_eff, self.x0

        

        if self.physics.coupling=='linear' and self.physics.D==3:
            # charge = scalar force / yukawa force, from analytic solution

            if rho_eff * x0**2 > 2.*(1.-phi_m):
                old_charge = 1. - (1. - 2.*(1.-phi_m)/(rho_eff * x0**2))**1.5
            else:
                old_charge = 1.
                
            # we define a constant function this way so that it has information about the function space
            self.old_charge = d.interpolate( d.Constant(old_charge), self.fem.S )

            
        elif self.physics.coupling=='linear' and self.physics.D==2:
            
            expression_A = 4.*(self.phi_m-1.)/(self.rho_eff * self.x0**2)
            expression_B = 1. + np.sqrt(2.)* k0(np.sqrt(2.)*x0)/(x0*k1(np.sqrt(2.)*x0))
            
            if expression_B + expression_A < 0.:
                old_charge = 1.
            else:
                old_charge = 1. - np.exp(expression_B + \
                                       lambertw(-np.exp(-expression_B)*(expression_A+expression_B),k=-1))
                
            # we define a constant function this way so that it has information about the function space
            self.old_charge = d.interpolate( d.Constant(old_charge), self.fem.S )


             
        elif self.physics.coupling=='quadratic' and self.physics.D==3:
            m = self.nu_eff
            ms = 4. * np.pi * rho_eff * x0**3 / 3. # source mass in natural units
            
            # "old_lamA" is the analytic expression for the screening factor
            # when phi*grad(phi) is approximated as vev*grad(phi)
            old_lamA = 4. * np.pi * (1. + np.sqrt(2.)*x0) * (1. - phi_m) \
                       * self.physics.Vev**2/(self.physics.mu*self.source.Ms) \
                              * (m*x0 - np.tanh(m*x0))/(m + np.sqrt(2.)*np.tanh(m*x0))
            
            # scalar force includes factor of phi, not just grad(phi)
            old_phi = d.Function( self.fem.S )
            if self.phi_at_infty=='zero':
                old_phi.vector()[:] = 1. + self.guess.vector()[:] / self.vev
            elif self.phi_at_infty=='vev':
                old_phi.vector()[:] = self.guess.vector()[:] / self.vev
            
            self.old_charge = d.Function( self.fem.S )
            self.old_charge.vector()[:] = old_lamA * old_phi.vector()[:]



        elif self.physics.coupling=='quadratic' and self.physics.D==2:
            m = self.nu_eff
            ms = np.pi * rho_eff * x0**2 # source mass in natural units

            # "old_lamA" is the analytic expression for the screening factor
            # when phi*grad(phi) is approximated as vev*grad(phi)
            if m*x0 > 1e2:
                # for large arguments, i1 overflows. This asymptotic series for i0(x)/i1(x)
                # converges to the real answer within machine precision for x>~100
                i0_by_i1 = lambda x : 1. + 1./(2*x) + 3./(8.*x**2) + 3. / (8.*x**3) + 63./( 128. * x**4 ) + \
                           27./( 32. * x**5 ) + 1899. / (1024.*x**6) + 81./(16.*x**7) + 543483. / (32768. * x**8 )

                old_lamA = 2. * np.pi * np.sqrt(2.) * x0 * m * (1. - phi_m) * k1(np.sqrt(2.)*x0) \
                           * self.physics.Vev**2/self.source.Ms / \
                           ( m*k0(np.sqrt(2.)*x0) + np.sqrt(2.)*k1(np.sqrt(2.)*x0) * i0_by_i1(m*x0) )
            else:
                old_lamA = 2. * np.pi * np.sqrt(2.) * x0 * m * (1. - phi_m) * k1(np.sqrt(2.)*x0) * i1(m*x0) \
                           * self.physics.Vev**2/self.source.Ms / \
                           ( m*i1(m*x0)*k0(np.sqrt(2.)*x0) + np.sqrt(2.)*i0(m*x0)*k1(np.sqrt(2.)*x0) )
                
            # scalar force includes factor of phi, not just grad(phi)
            old_phi = d.Function( self.fem.S )
            if self.phi_at_infty=='zero':
                old_phi.vector()[:] = 1. + self.guess.vector()[:] / self.vev
            elif self.phi_at_infty=='vev':
                old_phi.vector()[:] = self.guess.vector()[:] / self.vev
            
            self.old_charge = d.Function( self.fem.S )
            self.old_charge.vector()[:] = old_lamA * old_phi.vector()[:]

