# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver


from fem import Fem

import numpy as np
import dolfin as d

from dolfin import Expression, grad, inner, dx

from scipy.special import iv, i0, i1, jv, j0, j1, kv, k0, k1, yv
from scipy.optimize import newton
from scipy.optimize import brentq, fminbound
from scipy.optimize import fsolve
from scipy.optimize import root


class InitialGuess(object):
    
    def __init__( self, fem, source, physics,
                  mn=None, mf=None, phi_at_infty='zero', user_set_part=None ):
                

        # finite element properties, including the mesh
        self.fem = fem
        
        # source properties
        self.source = source

        # parameters of the theory, including masses, coupling and self-coupling
        self.physics = physics

        # rescalings (in units Planck mass)
        # distance rescaling
        self.mn = mn
        if self.mn is None:
            self.mn = self.source.rs / self.source.Rs
        # field rescaling
        self.mf = mf
        if self.mf is None:
            self.mf = self.physics.Vev
            
        # value of field at infinity (i.e. do we take phi or (phi - vev) ?)
        self.phi_at_infty = phi_at_infty

        # rescaled vev
        self.vev = self.physics.Vev/self.mf

        # source density and dimensionless source radius
        self.rho = self.physics.Rho_bar
        self.x0 = self.physics.x0

        # derived params
        self.rho_eff, self.nu, self.nu2_eff, self.nu_eff = None, None, None, None
        self.switch_b, self.switch_c, self.phi_c = None, None, None
        self.phi_m, self.a1 = None, None
        self.high_density, self.medium_density = None, None
        self.compute_derived_params()

        # N-part solution
        self.user_set_part = user_set_part
        self.two_parts, self.three_parts, self.four_parts = None, None, None
        self.x1, self.x2 = None, None

        if self.user_set_part is None:
            # choose appropriate analytical approximation based on the physical parameters
            self.choose_n_part_solution()
        else:
            # choose the user-set analytical approximation irrespectively of the physical parameters
            if self.user_set_part==2:
                self.x1, self.x2, self.two_parts = self.get_two_parts()
                self.two_parts = True
            elif self.user_set_part==3:
                self.x1, self.x2, self.three_parts = self.get_three_parts()
                self.three_parts = True
            elif self.user_set_part==4:
                self.x1, self.x2, self.four_parts = self.get_four_parts()
                self.four_parts = True
            else:
                message = "Invalid choice of analytic approximation. Valid choices are: None, 2, 3, 4"
                raise ValueError, message
                
        

        # compute initial guess
        self.guess = None # rescaled units (as used by the code)
        self.Guess = None # physical units
        self.varPhi = None # with correct vev, for plotting
        self.solve() # assign initial guess here






    def compute_derived_params( self ):

        rho, lam = self.rho, self.physics.lam
        mu, M = self.physics.mu, self.physics.M

        if self.physics.coupling=='quadratic':
            self.rho_eff  = rho / (mu*M)**2 # dimensionless
            
            if self.rho_eff > 1.: # screened regime
                self.nu = np.sqrt(self.rho_eff - 1.)
                self.phi_m = 0.
                
            elif self.rho_eff < 1.: # unscreened regime
                self.nu = np.sqrt(2.*(1. - self.rho_eff)) 
                self.phi_m = np.sqrt(1. - self.rho_eff)
                
            else:
                # the case rho_eff = 1 can't be obtained analytically
                # for this case, we set nu = phi_m = 1e-10 artificially for the purpose 
                # of obtaining an initial guess for the Newton algorithm
                self.nu = 1e-10
                self.phi_m = 1e-10
                message = "This problem is on the resonance rho = (mu * M)^2: the analytic approximations fail in this regime.\n" + \
                    "Now setting nu = phi_m = 1e-10 for the purpose of obtaining an initial guess"
                print(message)
                
            self.nu2_eff = self.nu**2
            self.nu_eff = self.nu
            self.switch_b = self.nu**2 * self.phi_m
            self.switch_c = 0.
            self.phi_c = self.phi_m

            
        
        elif self.physics.coupling=='linear':
            self.rho_eff = np.sqrt(lam) * rho / ( mu**3 * M ) # dimensionless
            self.high_density = ( self.rho_eff > 1. )
            self.medium_density = ( self.rho_eff > np.sqrt(12.)/9. ) and (not self.high_density)
            
            # approximation valid in limit as rho -> infinity
            self.phi_m = -self.rho_eff**(1./3.)
            
            if self.high_density:
                self.nu2_eff = -1.
                self.switch_b = 0.
                self.switch_c = 1.
                self.phi_c = self.rho_eff
                
            elif self.medium_density:
                self.nu2_eff = 2.
                self.nu_eff = np.sqrt(2.)
                self.switch_b = 2.
                self.switch_c = 1.
                self.phi_c = 1. - self.rho_eff/2.
       
            else:
                self.a1 = complex(-9.*self.rho_eff, np.sqrt(12. - 81. * self.rho_eff**2) )
                self.nu = np.sqrt( 1. + 2. * ((self.a1**2/12.)**(1./3.)).real ) # dimensionless
                self.phi_m = 2. * ((self.a1/18.)**(1./3.)).real # dimensionless
                
                self.nu2_eff = self.nu**2
                self.nu_eff = self.nu
                self.switch_b = self.nu**2*self.phi_m
                self.switch_c = 0.
                self.phi_c = self.phi_m
                
        else:
            message = 'wrong choice of coupling'
            raise ValueError, message

        





        
    def get_coeff_three_part_2D( self, full_output=False ):
        
        phi_c, x0, nu_eff = self.phi_c, self.x0, self.nu_eff
        
        def nonlin_constraint(C1,x2):
            # nonlinear part of the system
            return x2/6. - C1/x2 + np.sqrt(2.)/6. * kv(1,np.sqrt(2)*x2) / kv(0,np.sqrt(2)*x2)
    
        def lin_constraint(x2):
            # linear part of the system
            if self.physics.coupling=='linear' and self.high_density:
                M = np.array([ [ -j0(x0), np.log(x0), 1., ],
                               [ j1(x0), 1./x0, 0. ],
                               [ 0., -np.log(x2), -1. ]])
            
            else:
                M = np.array([[ -i0(nu_eff*x0), np.log(x0), 1. ],
                              [ -nu_eff*i1(nu_eff*x0), 1./x0, 0. ],
                              [ 0., -np.log(x2), -1. ]])
            b = np.array([ phi_c + x0**2/12., x0/6., -5./6. - x2**2/12. ])
            A, C1, C2 = np.linalg.solve(M, b)        
            return A, C1, C2
        
        def F(x2):
            # wrapper for the whole system - definition of residual function
            A,C1,C2 = lin_constraint(x2)
            return nonlin_constraint(C1,x2)
        
        try:
            x2 = newton( F, x0 ).real
            A,C1,C2 = lin_constraint(x2)
        except:
            # if solver diverges, the three-part solution may not be appropriate.
            # assign None to all params
            A, x2 = None, None
            C1, C2 = None, None
    
        if full_output:
            return A,C1,C2,x2
        else:
            return x2






        
    
    def get_coeff_three_part_3D( self, full_output=False ):

        phi_c, x0, nu_eff = self.phi_c, self.x0, self.nu_eff
        
        
        def nonlin_constraint(C1,x2):
            # nonlinear part of the system
            return x2/9. - C1/x2**2 + np.sqrt(2.)/6. + 1./(6.*x2)
        
        
        def lin_constraint(x2):
            # linear part of the system
            if self.physics.coupling=='linear' and self.high_density:
                M = np.array([[ -np.sin(x0)/x0, 1./x0, 1. ],
                              [ np.cos(x0) - np.sin(x0)/x0, 1./x0, 0. ],
                              [ 0., -1./x2, -1. ]])
            else:
                M = np.array([[ -np.sinh(nu_eff*x0)/x0, 1./x0, 1. ],
                              [ nu_eff*np.cosh(nu_eff*x0) - np.sinh(nu_eff*x0)/x0, 1./x0, 0. ],
                              [ 0., -1./x2, -1. ]])
            b = np.array([ phi_c + x0**2/18., -x0**2/9., -5./6. - x2**2/18. ])
            A, C1, C2 = np.linalg.solve(M, b)        
            return A, C1, C2

        
        def F(x2):
            # wrapper for the whole system - definition of residual function
            A,C1,C2 = lin_constraint(x2)
            return nonlin_constrant(C1,x2)
    
        
        try:
            x2 = newton( F, x0 ).real
            A, C1, C2 = lin_constraint(x2)
        except:
            # if solver diverges, the three-part solution may not be appropriate.
            # assign None to all params
            A, x2 = None, None
            C1, C2 = None, None
    
    
        if full_output:
            return A,C1,C2,x2
        else:
            return x2
    




        
    def get_coeff_four_part_2D( self, full_output=False ):
        
        phi_c, x0, nu_eff = self.phi_c, self.x0, self.nu_eff
        nu2_eff = self.nu2_eff
    
        def nonlin_constraints(C1,C2,x1,x2):
            # nonlinear part of the system
            return [ x2/6. - C1/x2 + np.sqrt(2.)/6. * kv(1,np.sqrt(2)*x2) / kv(0,np.sqrt(2)*x2), \
                     -1./3. - x1**2/12. + C1*np.log(x1) + C2 ]
    
        def lin_constraints(x1,x2):
            # linear part of the system
            if self.physics.coupling=='linear' and self.high_density:
                M = np.array([ [ -j0(x0)*yv(0,x1), j0(x0)*yv(0,x1) - j0(x1)*yv(0,x0), 0., 0. ],
                               [ j1(x0)*yv(0,x1), -j1(x0)*yv(0,x1) + j0(x1)*yv(1,x0), 0., 0. ],
                               [ 0., j1(x1)*yv(0,x1) - j0(x1)*yv(1,x1) , yv(0,x1)/x1, 0. ],
                               [ 0., 0., -np.log(x2), -1. ]])
        
            else:
                M = np.array([ [ -i0(nu_eff*x0)*yv(0,x1), j0(x0)*yv(0,x1) - j0(x1)*yv(0,x0), 0., 0. ],
                               [ -nu_eff*i1(nu_eff*x0)*yv(0,x1), -j1(x0)*yv(0,x1) + j0(x1)*yv(1,x0), 0., 0. ],
                               [ 0., j1(x1)*yv(0,x1) - j0(x1)*yv(1,x1) , yv(0,x1)/x1, 0. ],
                               [ 0., 0., -np.log(x2), -1. ]])
            
            b = np.array([ phi_c*yv(0,x1) - (1./3.)*yv(0,x0), (1./3.)*yv(1,x0), \
                           x1*yv(0,x1)/6. - (1./3.)*yv(1,x1), -5./6. - x2**2/12. ])

            # get A, B, C1, C2
            A, B, C1, C2 = np.linalg.solve(M, b)        
            return A, B, C1, C2
    
        def F(X):
            # wrapper for the whole system - definition of residual function
            x1, x2 = X
            A,B,C1,C2 = lin_constraints(x1,x2)
            return nonlin_constraints(C1,C2,x1,x2)

        x1, x2 = root( F, [x0, x0] ).x
    
        if full_output:
            A,B,C1,C2 = lin_constraints(x1,x2)
            return A,B,C1,C2,x1,x2
        else:
            return x1,x2
    
    





       
    def get_coeff_four_part_3D( self, full_output=False ):

        phi_c, x0, nu_eff = self.phi_c, self.x0, self.nu_eff
    
        def nonlin_constraints(C1,C2,x1,x2):
            # nonlinear part of the system
            return [ x2/9. - C1/x2**2 + np.sqrt(2.)/6. + 1./(6.*x2), \
                     -1./3. - x1**2/18. + C1/x1 + C2 ]
    
        def lin_constraints(x1,x2):
            # linear part of the system
            if self.physics.coupling=='linear' and self.high_density:
                M = np.array([ [ -np.sin(x0), np.sin(x0) - np.sin(x1)*np.cos(x0)/np.cos(x1), 0., 0. ],
                               [ np.sin(x0) - x0*np.cos(x0), \
                                 x0*np.cos(x0) - np.sin(x0) + np.sin(x1)*(np.cos(x0) + x0*np.sin(x0))/np.cos(x1), 0., 0. ],
                               [ 0., -x1*np.cos(x1) - x1*np.sin(x1)*np.tan(x1), -1., 0. ],
                               [ 0., 0., -1./x2, -1. ]])
            else:
                M = np.array([ [ -np.sinh(nu_eff*x0), np.sin(x0) - np.sin(x1)*np.cos(x0)/np.cos(x1), 0., 0. ],
                               [ np.sinh(nu_eff*x0) - nu_eff*x0*np.cosh(nu_eff*x0), \
                                 x0*np.cos(x0) - np.sin(x0) + np.sin(x1)*(np.cos(x0) + x0*np.sin(x0))/np.cos(x1), 0., 0. ],
                               [ 0., -x1*np.cos(x1) - x1*np.sin(x1)*np.tan(x1), -1., 0. ],
                               [ 0., 0., -1./x2, -1. ]])
            b = np.array([ x0*phi_c - (x1/3.)*np.cos(x0)/np.cos(x1), (x1/3.)*(np.cos(x0) + x0*np.sin(x0))/np.cos(x1), \
                              x1**3/9. - x1/3. * (1. + x1*np.tan(x1)), -5./6. - x2**2/18. ])
            # get A, B, C1, C2
            A, B, C1, C2 = np.linalg.solve(M, b)        
            return A, B, C1, C2
    
        def F(X):
            # wrapper for the whole system - definition of residual function
            x1, x2 = X
            A,B,C1,C2 = lin_constraints(x1,x2)
            return nonlin_constraints(C1,C2,x1,x2)
    
        x1, x2 = root( F, [x0, x0] ).x
    
        if full_output:
            A,B,C1,C2 = lin_constraints(x1,x2)
            return A,B,C1,C2,x1,x2
        else:
            return x1,x2





    def get_two_parts( self ):

        # get rid of x1 and x2 as they're not needed 
        x1 = -1.
        x2 = -1.
        
        if self.physics.D==3:

            if self.physics.coupling=='linear' and self.high_density:

                # coefficient for two-part solution
                A = (1. + np.sqrt(2.)*self.x0 ) * (1.-self.phi_c)/(np.cos(self.x0)+np.sqrt(2.)*np.sin(self.x0))
                # test for validity of two-part solution
                two_parts = ( self.phi_c + A * np.sin(self.x0)/self.x0 > 5./6. )

            else:
                A = (1. + np.sqrt(2.)*self.x0 ) * (1.-self.phi_c)/(self.nu_eff*np.cosh(self.nu_eff*self.x0)+ \
                                                                   np.sqrt(2.)*np.sinh(self.nu_eff*self.x0))
                
                two_parts = ( self.phi_c + A * np.sinh(self.nu_eff*self.x0)/self.x0 > 5./6. )

        elif self.physics.D==2:

            if self.physics.coupling=='linear' and self.high_density:
                A = np.sqrt(2.)*(1.-self.phi_c)*k1(np.sqrt(2.)*self.x0)/ \
                    (-j1(self.x0)*k0(np.sqrt(2.)*self.x0)+np.sqrt(2.)*j0(self.x0)*k1(np.sqrt(2.)*self.x0))
                two_parts = ( self.phi_c + A * j0(self.x0) > 5./6. )

            else:

                A = np.sqrt(2.)*(1.-self.phi_c)*k1(np.sqrt(2.)*self.x0)/ \
                    (self.nu_eff*i1(self.nu_eff*self.x0)*k0(np.sqrt(2.)*self.x0)+\
                     np.sqrt(2.)*i0(self.nu_eff*self.x0)*k1(np.sqrt(2.)*self.x0))
                # test for validity of two-part solution
                two_parts = ( self.phi_c + A * i0(self.nu_eff*self.x0) > 5./6. )


        return x1, x2, two_parts
                

            

        



    def get_three_parts( self ):

        # get rid of x1 as it's not needed
        x1 = -1.

        if self.physics.D==3:
            
            A, _, _, x2 = self.get_coeff_three_part_3D( full_output=True )

            if x2 is None:
                # the search for the three-part coefficients did not converge:
                # the three-part solution may not be appropriate
                three_parts_lower = False
                three_parts_upper = False

            else:
            
                # tests for validity of three-part solution
                if self.physics.coupling=='linear' and self.high_density:
                    three_parts_lower = ( self.phi_c + A * np.sin(self.x0)/self.x0 > 1./3. )
                    three_parts_upper = ( 5./6. > self.phi_c + A * np.sin(self.x0)/self.x0 )
                
                else:
                    three_parts_lower = ( self.phi_c + A * np.sinh(self.nu_eff*self.x0)/self.x0 > 1./3. )
                    three_parts_upper = ( 5./6. > self.phi_c + A * np.sinh(self.nu_eff*self.x0)/self.x0 )
                

        elif self.physics.D==2:
            
            A, _, _, x2 = self.get_coeff_three_part_2D( full_output=True )

            if x2 is None:
                # the search for the three-part coefficients did not converge:
                # the three-part solution may not be appropriate
                three_parts_lower = False
                three_parts_upper = False

            else:
                
                if self.physics.coupling=='linear' and self.high_density:
                    three_parts_lower = ( self.phi_c + A * j0(self.x0) > 1./3. )
                    three_parts_upper = ( 5./6. >  self.phi_c + A * j0(self.x0) )
                    
                else:
                    three_parts_lower = ( self.phi_c + A * i0(self.nu_eff*self.x0) > 1./3. )
                    three_parts_upper = ( 5./6. > self.phi_c + A * i0(self.nu_eff*self.x0) )


        three_parts = three_parts_lower and three_parts_upper

        return x1, x2, three_parts



    

    def get_four_parts( self ):

        if self.physics.D==3:
            
            A, _, _, _, x1, x2 = self.get_coeff_four_part_3D( full_output=True )
            
            if self.physics.coupling=='linear' and self.high_density:
                four_parts = ( 1./3. > self.phi_c + A * np.sinh(self.x0)/self.x0 )
            else:
                four_parts = ( 1./3. > self.phi_c + A * np.sinh(self.nu_eff*self.x0)/self.x0 )

        elif self.physics.D==2:
            
            A, _, _, _, x1, x2 = self.get_coeff_four_part_2D( full_output=True )
            
            if self.physics.coupling=='linear' and self.high_density:
                four_parts = ( 1./3. > self.phi_c + A * j0(self.x0) )
            else:
                four_parts = ( 1./3. > self.phi_c + A * i0(self.nu_eff*self.x0) )

        return x1, x2, four_parts






    def choose_n_part_solution( self ):

        # try the two parts solution
        self.x1, self.x2, self.two_parts = self.get_two_parts()

        # if it fails, try the three or four part solution
        if not self.two_parts:
            self.x1, self.x2, self.three_parts = self.get_three_parts()

            if not self.three_parts:
                self.x1, self.x2, self.four_parts = self.get_four_parts()



    
    
    
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


        
    def solve( self ):

        # all rescaled units
        
        # define the mass as an expression, for the three regimes
        code_m2 = '''
        class m2 : public Expression {              
            public:
        
            double mu, mn, x2, x1, x0;
            double nu2_eff; // -1 for linear coupling and high-density sources
                            // 2 for linear coupling and intermediate-density sources
                            // nu^2 otherwise

            m2() : Expression() {}

            void eval(Array<double>& values, const Array<double>& x) const {
                if ( x[0]*mu/mn < x0 ) { values[0] = nu2_eff * pow(mu/mn,2) ; }
                else if ( x[0]*mu/mn < x1 ) { values[0] = -1. * pow(mu/mn,2) ; }
                else if ( x[0]*mu/mn < x2 ) { values[0] = 0. ; }
                else { values[0] = 2. * pow(mu/mn,2) ; }
            }

        };'''
        
        m2 = d.Expression( code_m2, degree=self.fem.func_degree, mu=self.physics.mu, mn=self.mn,
                           x2=self.x2, x1=self.x1, x0=self.x0, nu2_eff=self.nu2_eff )
    
        code_b = '''
        class b : public Expression {              
            public:

            double mu, mn, x2, x1, x0, vev;
            double switch_b; // 0 for linear coupling with high-density sources
                             // 2 for linear coupling with intermediate-density sources
                             // nu**2*phi_m otherwise

            b() : Expression() {}

            void eval(Array<double>& values, const Array<double>& x) const {
                if ( x[0]*mu/mn < x0 ) { values[0] = - switch_b * pow(mu/mn,2) * vev ; }
                else if ( x[0]*mu/mn < x1 ) { values[0] = 0 ; }
                else if ( x[0]*mu/mn < x2 ) { values[0] = -1./3. * pow(mu/mn,2) * vev ; }
                else { values[0] = -2. * pow(mu/mn,2) * vev ; }
            }
                     
        };'''
    
        b = d.Expression( code_b, degree=self.fem.func_degree, mu=self.physics.mu, mn=self.mn,
                          x2=self.x2, x1=self.x1, x0=self.x0, vev=self.vev, switch_b=self.switch_b )
        
        code_c = "x[0]*mu/mn < x0 ? switch_c / ( M * mf ) * pow(mn,D-2) : 0."
        c = d.Expression( code_c, degree=self.fem.func_degree, mu=self.physics.mu, mn=self.mn,
                          M=self.physics.M, x0=self.x0, switch_c=self.switch_c, mf=self.mf, D=self.physics.D )
  
        
        # trial and test function
        phi = d.TrialFunction( self.fem.S )
        v = d.TestFunction( self.fem.S )
        
        # r^(D-1)
        rD = Expression('pow(x[0],D-1)', D=self.physics.D, degree=self.fem.func_degree)

        # get Dirichlet boundary conditions
        bc_phi = self.get_Dirichlet_bc()
        
        if self.phi_at_infty=='vev':
            a = - inner( grad(phi), grad(v) ) * rD * dx - m2 * phi * v * rD * dx
            L = ( b + c * self.source.rho ) * v * rD * dx
        elif self.phi_at_infty=='zero':
            a = - inner( grad(phi), grad(v) ) * rD * dx - m2 * phi * v * rD * dx
            L = ( b + m2 * self.vev + c * self.source.rho ) * v * rD * dx
            
        # define a vector with the solution
        sol = d.Function( self.fem.S )
        
        # solve linearised system
        pde = d.LinearVariationalProblem( a, L, sol, bc_phi )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()

        # assign solution - rescaled units (used in the code)
        self.guess = sol
        # physical units
        self.Guess = d.Function( self.fem.S )
        self.Guess.vector()[:] = sol.vector()[:] * self.mf

        # for plotting
        if self.phi_at_infty=='zero':
            self.varPhi = d.Function( self.fem.S )
            self.varPhi.vector()[:] = self.Guess.vector()[:] + self.physics.Vev
        else:
            self.varPhi = self.Guess.copy()
        
        
        
        
        
      
