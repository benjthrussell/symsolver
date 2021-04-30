# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver

# modified from phienics - https://github.com/scaramouche-00/phienics

import dolfin as d

class Fem(object):

    def __init__( self, mesh, func_cont='CG', func_disc='DG', func_degree=None ):

        # mesh
        self.mesh = mesh

        # interpolating functions
        if func_degree is None:
            if ( self.mesh.physics.D==3 and self.mesh.physics.x0 < 3e-3 ):
                func_degree = 6
            else:
                func_degree = 5

        self.func_cont = func_cont
        self.func_disc = func_disc
        self.func_degree = func_degree

        # continuous function space for single scalar function
        self.Pn = d.FiniteElement( self.func_cont, d.interval, self.func_degree )
        self.S = d.FunctionSpace( self.mesh.mesh, self.Pn )

        # discontinuous function space for single scalar function
        self.dPn = d.FiniteElement( self.func_disc, d.interval, self.func_degree ) 
        self.dS = d.FunctionSpace( self.mesh.mesh, self.dPn )


