"""
.. module:: symmetron
.. synopsys:: interface to the available solvers
"""

# Copyright (C) Benjamin Thrussell, Daniela Saadeh 2021
# This file is part of symsolver


from quadratic_zero import QuadraticZero
from quadratic_vev import QuadraticVev
from linear_zero import LinearZero
from linear_vev import LinearVev


def SymSolver( fem, source, physics,
               mn=None, mf=None, phi_at_infty='zero',
               abs_dphi_tol=1e-16, rel_dphi_tol=1e-16, abs_res_tol=1e-10, rel_res_tol=1e-20,
               max_iter=50, criterion='residual', norm_change='linf', norm_res='linf' ):

    
    if physics.coupling=='quadratic' and phi_at_infty=='vev':
        abs_rel_tol = None # this sets the tolerance within the child class
        solver = QuadraticVev( fem, source, physics,
                               mn=mn, mf=mf,
                               abs_dphi_tol=abs_dphi_tol, rel_dphi_tol=rel_dphi_tol,
                               abs_res_tol=abs_res_tol, rel_res_tol=rel_res_tol,
                               max_iter=max_iter, criterion=criterion,
                               norm_change=norm_change, norm_res=norm_res )
        
    elif physics.coupling=='quadratic' and phi_at_infty=='zero':
        solver = QuadraticZero( fem, source, physics,
                                mn=mn, mf=mf,
                                abs_dphi_tol=abs_dphi_tol, rel_dphi_tol=rel_dphi_tol,
                                abs_res_tol=abs_res_tol, rel_res_tol=rel_res_tol,
                                max_iter=max_iter, criterion=criterion,
                                norm_change=norm_change, norm_res=norm_res )
        
    elif physics.coupling=='linear' and phi_at_infty=='vev':
        solver = LinearVev( fem, source, physics,
                            mn=mn, mf=mf,
                            abs_dphi_tol=abs_dphi_tol, rel_dphi_tol=rel_dphi_tol,
                            abs_res_tol=abs_res_tol, rel_res_tol=rel_res_tol,
                            max_iter=max_iter, criterion=criterion,
                            norm_change=norm_change, norm_res=norm_res )
        
    elif physics.coupling=='linear' and phi_at_infty=='zero':
        solver = LinearZero( fem, source, physics,
                             mn=mn, mf=mf, 
                             abs_dphi_tol=abs_dphi_tol, rel_dphi_tol=rel_dphi_tol,
                             abs_res_tol=abs_res_tol, rel_res_tol=rel_res_tol,
                             max_iter=max_iter, criterion=criterion,
                             norm_change=norm_change, norm_res=norm_res )

    return solver
