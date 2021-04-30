# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

from utils import project, rD_norm, get_values
from initial_guess import InitialGuess
from old_analytics import OldAnalytics

from dolfin import inner, grad, Expression, Constant, dx
from scipy.optimize import brentq, fminbound

import numpy as np
import dolfin as d



        


class Solver(object):

    def __init__( self, fem, source, physics,
                  mn=None, mf=None, phi_at_infty='zero',
                  abs_dphi_tol=1e-16, rel_dphi_tol=1e-16, abs_res_tol=1e-10, rel_res_tol=1e-10,
                  max_iter=50, criterion='residual', norm_change='linf', norm_res='linf' ):

        

        # finite element properties, including the mesh
        self.fem = fem
        
        # source properties
        self.source = source

        # parameters of the theory, including masses, coupling and self-coupling
        self.physics = physics

        # rescalings (in units Planck mass)
        # distance rescaling
        if mn is None:
            self.mn = self.source.rs / self.source.Rs
        else:
            self.mn = mn
        # field rescaling
        if mf is None:
            self.mf = self.physics.Vev
        else:
            self.mf = mf
            
        # value of field at infinity (i.e. do we take phi or (phi - vev) ?)
        self.phi_at_infty = phi_at_infty
        
        # tolerance of the non-linear solver - change in solution (dphi)
        self.abs_dphi_tol = abs_dphi_tol
        self.rel_dphi_tol = rel_dphi_tol
        # tolerance of the non-linear solver - residual (F)
        self.abs_res_tol = abs_res_tol
        self.rel_res_tol = rel_res_tol   
        # maximum number of iterations
        self.max_iter = max_iter
        # choice of convergence criterion and norm
        self.criterion = criterion
        self.norm_change = norm_change
        self.norm_res = norm_res

        # flag: has it converged?
        self.converged = False
        # iteration counter
        self.i = 0
       
        # absolute and relative error at every iteration
        # change in solution
        self.abs_dphi = np.zeros( self.max_iter )
        self.rel_dphi = np.zeros( self.max_iter )
        # residual
        self.abs_res = np.zeros( self.max_iter )
        self.rel_res = np.zeros( self.max_iter )
        
        # initial residual and norm of solution at initial iteration - used
        # for relative change and residual; set in the update_errors function
        self.phi0_norm = None
        self.F0_norm = None

        # solution and field profiles (computed by the solver)
        self.phi = None
        self.Phi = None
        self.varPhi = None # field with correct Vev

        # field gradient
        self.grad_Phi = None
        
        # healing length
        # definition: percentage of the vev
        self.healing_threshold = 0.95
        self.r_healing = None # units source radius
        self.R_healing = None # physical units

        # initial guess
        self.initial_guess = InitialGuess( self.fem, self.source, self.physics, \
                                  self.mn, self.mf, self.phi_at_infty )
        self.old_analytics = OldAnalytics( self.fem, self.source, self.physics, \
                                 self.mn, self.mf, self.phi_at_infty )





    # theory-dependent functions
    def weak_residual_form( self, sol ):
        pass
    
    def strong_residual_form( self, sol, norm='linf'):
        pass
    
    def linear_solver( self, phi_k ):
        pass
    
    def scalar_force( self ):
        pass
    
    def compute_derrick( self ):
        pass
    
    def compute_yukawa_force( self ):
        pass
    
    def compute_screening_factor( self ):
        pass
    
    
    
    
    def get_Dirichlet_bc( self ):

        # define values at infinity
        # for 'infinity', we use the last mesh point, i.e. r_max (i.e. mesh[-1])
        
        if self.phi_at_infty=='vev':
            vev = self.physics.Vev / self.mf # rescaled vev
            phiD = d.Constant( vev )
        elif self.phi_at_infty=='zero':
            phiD = d.Constant( 0. )

        # define 'infinity' boundary: the rightmost mesh point - within machine precision
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS

        bc_phi = d.DirichletBC( self.fem.S, phiD, boundary, method='pointwise' )

        return bc_phi
    
    
    
    
    
    
    def compute_healing_length( self ):

        threshold = self.healing_threshold
        if self.phi_at_infty=='vev':
            delta_Phi = self.physics.Vev - self.Phi(0)
        elif self.phi_at_infty=='zero':
            delta_Phi = - self.Phi(0)
        
        # get bracket for the healing length:
        # mesh points to the left and right of the true healing length
        r_values, Phi_values = get_values( self.Phi, output_mesh=True )
        idx = np.where( Phi_values > self.Phi(0) + threshold * delta_Phi )[0][0]
        left = r_values[idx-1]
        right = r_values[idx]

        # now find the healing length in that bracket using the Brent method
        F = lambda r : self.Phi(r) - self.Phi(0) - threshold * delta_Phi
        try:
            r_healing = brentq( F, left, right )
        except:
            r_healing = np.nan

        # rescaled and physical
        self.r_healing = r_healing
        self.R_healing = r_healing / self.mn
    


    
    def compute_yukawa_force( self ):

        # solve an equation of motion that will give you
        # the force from a scalar without nonlinearities

        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        mu = Constant( self.physics.mu )
        mn = Constant( self.mn )
        
        # trial and test function
        phi = d.TrialFunction( self.fem.S )
        v = d.TestFunction( self.fem.S )
        
        # boundary condition - always zero 
        phiD = d.Constant( 0. )
        # define 'infinity' boundary: the rightmost mesh point - within machine precision
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS
        Dirichlet_bc = d.DirichletBC( self.fem.S, phiD, boundary, method='pointwise' )

        # r^(D-1)
        rD = Expression('pow(x[0],D-1)', D=self.physics.D, degree=self.fem.func_degree)
        
        # m^2 = 2.
        a = - inner( grad(phi), grad(v) ) * rD * dx - 2. * (mu/mn)**2 * phi * v * rD * dx
        L = mn**(self.physics.D-2.)*self.source.rho/(self.physics.M*self.mf) * v * rD * dx
        # the Yukawa potential has linear matter coupling even when
        # the symmetron has quadratic matter coupling
        
        yukawa = d.Function( self.fem.S )
        pde = d.LinearVariationalProblem( a, L, yukawa, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()

        self.yukawa = d.Function( self.fem.S )
        self.yukawa.vector()[:] = self.mf * yukawa.vector()[:]
    
    
    
    
    def postprocessing( self ):

        if self.Phi is None:
            message = "You haven't solved the field profile: please run solve() first."
            raise ValueError, message

        self.Phi_rs = self.varPhi( self.source.rs ) # field with correct vev
        self.grad_Phi_rs = self.grad_Phi( self.source.rs )
        self.grad_Phi_max = rD_norm( self.grad_Phi.vector(), self.physics.D, self.fem.func_degree, norm_type='linf' )
        self.compute_healing_length()
        self.compute_derrick()
        self.compute_screening_factor()
    
    
    
    
    
    
    def strong_residual( self, sol=None, units='rescaled', norm='linf' ):

        if sol is None:
            sol = self.phi

        F = self.strong_residual_form( sol, units )

        # 'none' = return function, not norm
        if norm=='none':
            result = F
        # from here on return a norm. This nested if is to preserve the structure of the original
        # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices
            result = rD_norm( F.vector(), self.physics.D, self.fem.func_degree, norm_type=norm )
        else:
            result = rD_norm( F, self.physics.D, self.fem.func_degree, norm_type=norm )

        return result
    
    
    
    
    
    
    def grad( self, field ):

        grad = Constant(self.mn) * field.dx(0)
        grad = project( grad, self.fem.dS, self.physics.D, self.fem.func_degree )

        return grad

    

    
    
    
    
    
    def compute_errors( self, dphi_k, phi_k ):
        
        # compute residual of solution at this iteration - assemble form into a vector
        F = d.assemble( self.weak_residual_form( phi_k ) )
        
        # ... and compute norm. L2 is note yet implemented
        if self.norm_res=='L2':
            message = "UV_ERROR: L2 norm not implemented for residuals. Please use a different norm ('l2' or 'linf')"
            raise L2_Not_Implemented_For_F, message
        
        F_norm = d.norm( F, norm_type=self.norm_res )
        
        # now compute norm of change in the solution
        # this nested if is to preserve the structure of the original built-in FEniCS norm function 
        # within the modified rD_norm function
        if self.norm_change=='L2':
            # if you are here, you are computing the norm of a function object, as an integral
            # over the whole domain, and you need the r^2 factor in the measure (i.e. r^2 dr )
            dphi_norm = rD_norm( dphi_k, self.physics.D, func_degree, norm_type=self.norm_change )
            
        else:
            # if you are here, you are computing the norm of a vector. For this, the built-in norm function is
            # sufficient. The norm is either linf (max abs value at vertices) or l2 (Euclidean norm)
            dphi_norm = d.norm( dphi_k.vector(), norm_type=self.norm_change )
            
        return dphi_norm, F_norm
    
    
    
    
    
    
    def solve( self ):

        phi_k = self.initial_guess.guess.copy(deepcopy=True)
        abs_dphi = d.Function( self.fem.S )

        while (not self.converged) and self.i < self.max_iter:

            # get solution at this iteration from linear solver
            sol = self.linear_solver( phi_k )
            
            # if this is the initial iteration, store the norm of the initial solution and initial residual 
            # for future computation of relative change and residual
            if self.i == 0:
                # the first 'sol' passed as input takes the place of what is normally the variation in the solution
                self.phi0_norm, self.F0_norm = self.compute_errors( sol, sol )

            # compute and store change in the solution
            abs_dphi.vector()[:] = sol.vector()[:] - phi_k.vector()[:]
            self.abs_dphi[self.i], self.abs_res[self.i] = self.compute_errors( abs_dphi, sol )
            # compute and store relative errors
            self.rel_dphi[self.i] = self.abs_dphi[self.i] / self.phi0_norm
            self.rel_res[self.i] = self.abs_res[self.i] / self.F0_norm


            # write report: to keep output legible only write tolerance for the criterion that's effectively working
            if self.criterion=='residual':
                print 'Non-linear solver, iteration %d\tabs_dphi = %.1e\trel_dphi = %.1e\t' \
                    % (self.i, self.abs_dphi[self.i], self.rel_dphi[self.i] ),
                print 'abs_res = %.1e (tol = %.1e)\trel_res = %.1e (tol = %.1e)' \
                    % ( self.abs_res[self.i], self.abs_res_tol, self.rel_res[self.i], self.rel_res_tol )

            else:
                print 'Non-linear solver, iteration %d\tabs_dphi = %.1e (tol = %.1e)\trel_dphi = %.1e (tol=%.1e)\t' \
                    % (self.i, self.abs_dphi[self.i], self.abs_dphi_tol, self.rel_dphi[self.i], self.rel_dphi_tol ),
                print 'abs_res = %.1e\trel_res = %.1e' \
                    % ( self.abs_res[self.i], self.abs_res_tol, self.rel_res[self.i], self.rel_res_tol )


            # check convergence
            if self.criterion=='residual':
                self.converged = ( self.abs_res[self.i] < self.rel_res_tol * self.F0_norm ) \
                or ( self.abs_res[self.i] < self.abs_res_tol )
                
            else:
                self.converged = ( self.abs_dphi[self.i] < self.rel_dphi_tol * self.phi0_norm ) \
                or ( self.abs_dphi[self.i] < self.abs_dphi_tol )

                
            # if maximum number of iterations has been reached without converging, throw a warning
            if ( self.i+1 == self.max_iter and ( not self.converged ) ):
                print "*******************************************************************************"
                print "   WARNING: the solver hasn't converged in the maximum number of iterations"
                print "*******************************************************************************"
            
            # update for next iteration
            self.i += 1
            phi_k.assign(sol)
            

        # pass the solution to the problem
        # rescaled units
        self.phi = sol
        # physical units
        self.Phi = d.Function( self.fem.S )
        self.Phi.vector()[:] = self.mf * self.phi.vector()[:]

        # for plotting
        if self.phi_at_infty=='zero':
            self.varPhi = d.Function( self.fem.S )
            self.varPhi.vector()[:] = self.Phi.vector()[:] + self.physics.Vev
        else:
            self.varPhi = self.Phi.copy()

        # gradient - physical
        self.grad_Phi = self.grad( self.Phi )

        # force - physical
        self.force = self.scalar_force()

        # get useful postprocessing quantities like the healing length
        # and tests from Derrick's theorem
        self.postprocessing()
