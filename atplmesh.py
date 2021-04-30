# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

# modified from phienics - https://github.com/scaramouche-00/phienics

from mesh import Mesh

import numpy as np
import dolfin as d





class ArcTanPowerLawMesh(Mesh):
    r"""
    This class implements the Arctan-power law mesh described in detail at Sec. 3.4.2 of
    <https://arxiv.org/abs/2011.07037>_ . This mesh applies the transformation:

    .. math:: r = T(x) = \frac{2}{\pi} r_{\rm s} \arctan{(kx)} + x^{\gamma}

    to obtain a nonlinear radial mesh (:math:`r` coordinate) starting from a uniform mesh (:math:`x` coordinate).

    *Arguments*
        r_min
            minimum radius (typically, 0.)
        r_max
            maximum radius box size (i.e. the simulation's 'infinity'). It must be large
            compared to the source radius and the Compton wavelength of the fields involved
        num_cells
            number of cells in the mesh
        too_fine
            this parameter is used to make sure points are not closer than machine precision
            allows to resolve. Points are required to be spaced at least
            :math:`\mathrm{too\_fine}\times\mathrm{DOLFIN\_EPS}`. See :func:`check_mesh`
        linear_refine
            after producing a non-uniform mesh, optionally slice cells in the 
            [linear\_start, linear\_stop] interval this many times (use 0 to skip this step)
        linear_start
            refine linearly from here
        linear_stop
            refine linearly up to here
        linear_cells
            if linear\_refine was chosen, how many cells were added by linear refinement
        r_rm
            optionally remove points around this radius using a transformation 
            :math:`r = A_{\rm rm}/2 \arctan{\left( k_rm x \right)}` (see section 3.4.3
            of the accompanying paper <this_will_be_the_arXiv_link>_ )
        A_rm
            transformation parameter; only valid if r_rm is a valid radiusx
        k_rm
            transformation parameter; only valid if r_rm is a valid radius
        k
            parameter :math:`k` in the :math:`T(x)` transformation
        gamma
            parameter :math:`gamma` in the :math:`T(x)` transformation

    """

    def __init__( self, physics, r_min=0., r_max=None,
                  num_cells=None, too_fine=1e4, Ntol=1e-8,
                  linear_refine=0, linear_start=None, linear_stop=None,
                  r_rm=None, A_rm=25., k_rm=20., adjust_transition=True,
                  k=None, gamma=None ):
        """The constructor"""

        if num_cells is None:
            extra = 0
            if physics.ms > 1. :
                extra = 50
                if np.log10(physics.x0) < -2:
                    extra += 50
            if physics.D==3:
                extra += 75
            num_cells = int( 200. - 50. * np.log10(physics.x0) + extra )
        
        Mesh.__init__( self, physics, r_min, r_max, num_cells, too_fine, Ntol,
                       linear_refine, linear_start, linear_stop,
                       r_rm, A_rm, k_rm, adjust_transition )
        
        # params of the non-linear transform
        if k is None:
            if ( physics.coupling=='quadratic' and physics.D==3 ):
                if np.log10(physics.x0) < -2.5:
                    k = 300.
                elif np.log10(physics.ms) > 2.5 :
                    k = 50.
                else:
                    k = 20.
            else:
                k = 20.
        self.k = k
                
        if gamma is None:
            if ( physics.D==3 and np.log10(physics.x0) < -2.5 ):
                gamma = 10.
            else:
                gamma = 11. - 0.5 * np.log10(physics.x0)
        self.gamma = gamma
        
        # build the mesh
        self.build_mesh()


        


    def check_params( self ):
        r"""
        Check that the input parameters make sense: :math:`k` and :math:`\gamma` must satisfy
        :math:`k>0` and :math:`\gamma \geq 1`.

        """

        gamma_k_test = ( self.k > 0. ) and ( self.gamma >= 1. )

        if not gamma_k_test:
            message = "Invalid mesh input parameters ( gamma, k ). It must be k>0 and gamma>=1."
            raise ValueError, message


      

    def baseline_transform( self ):
        r"""
        Defines the baseline trasformation

        .. math:: r = T(x) = \frac{2}{\pi} r_{\rm s} \arctan{(kx)} + x^{\gamma}

        together with its first and second derivatives, and an approximation for the inverse transformation
        at small and large radii.

        """

        k, gamma = self.k, self.gamma
        rs = self.rs
        
        # baseline transformation
        T = lambda x : rs * ( 2./np.pi * np.arctan( k * x ) + x**gamma )

        # derivatives of baseline transformation
        Tprime = lambda x : rs * ( 2./np.pi * k / ( 1. + (k*x)**2 ) + gamma * x**(gamma-1) )
        Tprimeprime = lambda x : rs * ( -4./np.pi * k**3 * x / ( 1. + (k*x)**2 )**2 + gamma * (gamma-1.) * x**(gamma-2) )
        
        # approximate inverse of baseline transformation for small and large r
        small_r_Tm1 = lambda r : r / Tprime( 0. ) 
        large_r_Tm1 = lambda r : abs( r/rs - 1. )**(1./gamma)

        return T, Tprime, Tprimeprime, small_r_Tm1, large_r_Tm1
