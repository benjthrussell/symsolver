# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver


from solver import Solver
from utils import project, rD_norm

from dolfin import inner, grad, Expression, Constant, dx
from dolfin import SubDomain
from scipy.optimize import brentq, fminbound

import dolfin as d
import numpy as np
import warnings


class QuadraticZero(Solver):

    
    def __init__( self, fem, source, physics,
                  mn=None, mf=None,
                  abs_dphi_tol=1e-16, rel_dphi_tol=1e-16, abs_res_tol=1e-10, rel_res_tol=1e-16,
                  max_iter=50, criterion='residual', norm_change='linf', norm_res='linf' ):
        

        # field rescaling - results in max size of terms in the equation ~ 1
        if mf is None:
            mf = physics.Vev * physics.Ms / physics.g / physics.Rs**(physics.D-2) / physics.M**2
          
        Solver.__init__( self, fem, source, physics,
                         mn, mf, 'zero',
                         abs_dphi_tol, rel_dphi_tol, abs_res_tol, rel_res_tol,
                         max_iter, criterion, norm_change, norm_res )
        
        # field value at source surface
        self.Phi_rs = None
        
        # gradient at source surface
        self.grad_Phi_rs = None
        
        # maximum gradient
        self.grad_Phi_max = None
        
        # tests from Derrick's theorem
        self.derrick = None
        
        # test of screening
        self.yukawa = None
        self.screening_factor = None

        # useful for plotting
        self.num_terms = 6 
    
    
    
    
    
    
    def strong_residual_form( self, sol, units ):

        if units=='rescaled':
            resc = 1.
        elif units=='physical':
            resc = self.mn**2 * self.mf
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError, message
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn, mf = Constant( self.mn ), Constant( self.mf )
        
        D = self.physics.D
        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        # I expand manually the Laplacian into (D-1)/r df/dr + d2f/dr2
        f = sol.dx(0).dx(0) + Constant(D-1.)/r * sol.dx(0) - lam*(mf/mn)**2*sol**3 \
            - 3.*d.sqrt(lam)*(mu*mf/mn**2)*sol**2 - 2.*(mu/mn)**2*sol - mn**(D-2.)/M**2*self.source.rho*sol \
            - (mu*mn**(D-2.)/(mf*M**2))*self.source.rho/d.sqrt(lam)
        f *= resc
        F = project( f, self.fem.dS, self.physics.D, self.fem.func_degree )
        
        return F
    
    
    
    
    
    
    def weak_residual_form( self, sol ):
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn, mf = Constant( self.mn ), Constant( self.mf )
        
        D = self.physics.D
        
        v = d.TestFunction( self.fem.S )
        
        # r^(D-1)
        rD = Expression('pow(x[0],D-1)', D=D, degree=self.fem.func_degree)
        
        # define the weak residual form
        F = - inner( grad(sol), grad(v)) * rD * dx - lam*(mf/mn)**2*sol**3 * v * rD * dx \
            - 3.*d.sqrt(lam)*(mu*mf/mn**2)*sol**2 * v * rD * dx - 2.*(mu/mn)**2*sol * v * rD * dx \
            - mn**(D-2.)/M**2*self.source.rho*sol * v * rD * dx \
            - (mu*mn**(D-2.)/(mf*M**2))*self.source.rho/d.sqrt(lam) * v * rD * dx
        
        return F
    
    
    
    
    
    
    def linear_solver( self, phi_k ):
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn, mf = Constant( self.mn ), Constant( self.mf )
        
        D = self.physics.D
        
        # boundary condition
        Dirichlet_bc = self.get_Dirichlet_bc()
        
        # trial and test function
        phi = d.TrialFunction( self.fem.S )
        v = d.TestFunction( self.fem.S )
        
        # r^(D-1)
        rD = Expression('pow(x[0],D-1)', D=D, degree=self.fem.func_degree)
        
        # bilinear form a and linear form L
        a = - inner( grad(phi), grad(v) ) * rD * dx + ( - 3.*lam*(mf/mn)**2*phi_k**2 \
            - 6.*d.sqrt(lam)*(mu*mf/mn**2)*phi_k - 2.*(mu/mn)**2 - (mn**(D-2.)/M**2)*self.source.rho ) \
            * phi * v * rD * dx
        L = ( (mu*mn**(D-2.)/(mf*M**2))*self.source.rho/d.sqrt(lam) - 2.*lam*(mf/mn)**2*phi_k**3 \
            - 3.*d.sqrt(lam)*(mu*mf/mn**2)*phi_k**2 ) * v * rD * dx
        
        # define a vector with the solution
        sol = d.Function( self.fem.S )
        
        # solve linearised system
        pde = d.LinearVariationalProblem( a, L, sol, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()
        
        return sol
    
    
    
    
    
    
    def scalar_force( self ):
        
        grad = self.grad( self.Phi )
        force = - (self.physics.Vev + self.Phi) * grad / Constant(self.physics.M**2)
        force = project( force, self.fem.dS, self.physics.D, self.fem.func_degree )
        
        return force
    
    
    


    def compute_screening_factor( self ):

        self.compute_yukawa_force()

        screening_factor = (self.physics.Vev + self.Phi) * self.Phi.dx(0) / (self.yukawa.dx(0) * self.physics.M)
        self.screening_factor = project( screening_factor, self.fem.dS, self.physics.D, self.fem.func_degree )
        
        



    def EoM_term( self, term, output_label=True ):
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn = Constant( self.mn )
        
        D = self.physics.D

        Phi = self.Phi
        varPhi = self.varPhi

        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )

        if term==1:
            Term = ( Constant(D-1.)/r * Phi.dx(0) + Phi.dx(0).dx(0) ) * mn**2
            label = r"$\nabla^2\phi$"
        elif term==2:
            Term = mu**2 * varPhi
            label = r"$m^2\phi$"
        elif term==3:
            Term = - lam * varPhi**3
            label = r"$-\lambda \phi^3$" 
        elif term==4:
            Term = mn**D/M**2 * self.source.rho * varPhi
            label = r"$\frac{\rho}{M^2}\phi$" 
        
        Term = project( Term, self.fem.dS, self.physics.D, self.fem.func_degree )
        
        if output_label:
            return Term, label
        else:
            return Term   



        
        
    
    def output_term( self, term='LHS', norm='none', units='rescaled', output_label=False ):
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn, mf = Constant( self.mn ), Constant( self.mf )
        
        D = self.physics.D
        
        if units=='rescaled':
            resc = 1.
            str_phi = '\\hat{\\varphi}'
            str_nabla2 = '\\hat{\\nabla}^2'
            str_m2 = '2\\left( \\frac{\\mu}{m_n} \\right)^2'
            str_lambdaphi3 = '\\lambda\\left(\\frac{m_f}{m_n}\\right)^2\\hat{\\phi}^3'
            str_phi2term = '3\\sqrt{\\lambda}\\left(\\frac{\\mu m_f}{m_n^2}\\right)\\varphi^2'
            str_rho = '\\frac{m_n^{D-2}}{M^2}\\hat{\\rho}'
            str_rhophi0 = '\\frac{\\mu m_n^{D-2}}{m_f M^2}\\frac{\\rho}{\\sqrt{\\lambda}}'
            
        elif units=='physical':
            resc = self.mn**2 * self.mf        
            str_phi = '\\varphi'
            str_nabla2 = '\\nabla^2'
            str_m2 = '2\\mu^2'
            str_lambdaphi3 = '\\lambda\\varphi^3'
            str_phi2term = '-3\\sqrt{\\lambda}\\mu\\varphi^2'
            str_rho = '\\frac{\\rho}{M^2}'
            str_rhophi0 = '\\frac{\\rho}{M^2 \\sqrt{\\lambda}}'
            
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError, message
        
        phi = self.phi
        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        if term=='LHS': # I expand manually the Laplacian into (D-1)/r df/dr + d2f/dr2
            Term = Constant(D-1.)/r * phi.dx(0) + phi.dx(0).dx(0) \
                 - lam*(mf/mn)**2*phi**3 - 3.*d.sqrt(lam)*(mu*mf/mn**2)*phi**2 \
                 - 2*(mu/mn)**2*phi - mn**(D-2.)/M**2*self.source.rho*phi
            label = r"$%s%s - %s - %s - %s%s - %s%s$" % \
            (str_nabla2, str_phi, str_lambdaphi3, str_phi2term, str_m2, str_phi, str_rho, str_phi)
        elif term=='RHS':
            Term = (mu*mn**(D-2.)/(mf*M**2))*self.source.rho/d.sqrt(lam)
            label = r"$%s$" % (str_rhophi0)
        elif term==1:
            Term = Constant(D-1.)/r * phi.dx(0) + phi.dx(0).dx(0)
            label = r"$%s%s$" % (str_nabla2, str_phi)
        elif term==2:
            Term = - lam*(mf/mn)**2*phi**3
            label = r"$-%s$" % (str_lambdaphi3)
        elif term==3:
            Term = - 3.*d.sqrt(lam)*(mu*mf/mn**2)*phi**2
            label = r"$-%s$" % (str_phi2term)
        elif term==4:
            Term = - 2*(mu/mn)**2*phi
            label = r"$-%s%s$" % (str_m2, str_phi)
        elif term==5:
            Term = -mn**(D-2.)/M**2*self.source.rho*phi
            label = r"$-%s%s$" % (str_rho, str_phi)
        elif term==6:
            Term = (mu*mn**(D-2.)/(mf*M**2))*self.source.rho/d.sqrt(lam)
            label = r"$%s$" % (str_rhophi0)
        # rescale if needed to get physical units
        Term *= resc
        
        Term = project( Term, self.fem.dS, self.physics.D, self.fem.func_degree )
        
        # 'none' = return function, not norm
        if norm=='none':
            result = Term
            # from here on return a norm. This nested if is to preserve the structure of the original
            # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices    
            result = rD_norm( Term.vector(), self.physics.D, self.fem.func_degree, norm_type=norm )
        else:
            result = rD_norm( Term, self.physics.D, self.fem.func_degree, norm_type=norm )
        
        if output_label:
            return result, label
        else:
            return result
    
    
    
    
    
    
    def compute_derrick( self ):
        
        D = self.physics.D
        
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        # r^(D-1)
        rD = Expression('pow(x[0],D-1)', D=D, degree=self.fem.func_degree)
        
        mu, M = Constant( self.physics.mu ), Constant( self.physics.M )
        lam = Constant( self.physics.lam )
        mn = Constant( self.mn )
        
        # integrate potential energy density
        eV = mu**2*self.Phi**2 + d.sqrt(lam)*mu*self.Phi**3 + lam/4.*self.Phi**4
        E_V = d.assemble( eV * rD * dx )
        E_V /= self.mn**D  # get physical distances - integral now has mass dimension 4 - D
        
        # integrate kinetic energy density
        eK = Constant(0.5) * self.grad_Phi**2
        E_K = d.assemble( eK * rD * dx )
        E_K /= self.mn**D # get physical distances - integral now has mass dimension 4 - D
        
        # matter coupling energy
        erho = self.source.rho * ( self.Phi + self.physics.Vev )**2 / ( 2. * M**2 )
        E_rho = d.assemble( erho * rD * dx ) # rescaled rho, and so the integral, has mass dimension 4 - D
        
        # integral terms of Derrick's theorem
        derrick1 = (D - 2.) * E_K + D*(E_V + E_rho)
        derrick4 = 2. * (D - 2.) * E_K
        
        # non-integral terms of Derrick's theorem - these have mass dimension 4 - D
        derrick2 = self.source.Rho_bar * self.source.Rs**D * \
                   ( self.Phi(self.fem.mesh.rs) + self.physics.Vev )**2 / \
                   (2. * self.physics.M**2 )
        derrick3 = self.source.Rho_bar * self.source.Rs**(D+1.) * \
                   ( self.Phi(self.fem.mesh.rs) + self.physics.Vev ) * self.grad_Phi(self.fem.mesh.rs) / self.physics.M**2
        
        self.derrick = [ derrick1, derrick2, derrick3, derrick4 ]
