# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tee-shaped coupler with fillet for Q components. 
"""

import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core.base import QComponent


class Round_Tee(QComponent):
    """The base `Round_Tee` class.

    Inherits `QComponent` class.

    Create a tee-shaped component with subtracted spacing. 
    This QComponent is separated from Qubits and other components, for simplicity when rendering to ANSYS.  

    Default Options:
        
    """

    default_options = Dict(
        ref_qc = None, rel_ori = "0", shift_x = "100um", shift_y = "100um", 
        cpw_width = "10um", cpw_gap = "6um", ext_len = "100um",
        tee_width = "5um", tee_length = "20um", tee_fillet = "10um", 
        helper = False,
        # 90 has dipole aligned along the +X axis,
        # while 0 has dipole aligned along the +Y axis
        )

    """Component metadata"""

    TOOLTIP = """The user component `Round_Tee` class."""

    def make(self):
        """Define the way the options are turned into QGeometry.

        The make function implements the logic that creates the geoemtry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed
        information, such as layer, subtract, etc.
        """
        p = self.p
        self.make_round_tee(p.cpw_width+2*p.cpw_gap, p.tee_fillet, p.tee_width+2*p.cpw_gap, subtract=1)
        self.make_round_tee(p.cpw_width            , p.tee_fillet, p.tee_width            , subtract=0)

    def make_round_tee(self, cpw_width, fillet, tee_width, subtract=False):
        p = self.p
        Q = self.p.ref_qc.options
        self.pos_x, self.pos_y = Q.pos_x, Q.pos_y
        
        # At 0 orientation, constructing tee from close-to-qubit part 
        rect_1 = draw.rectangle(tee_width, p.tee_length) # closest-to-qubit part 
        rect_2 = draw.rectangle(fillet, cpw_width + 2*fillet) # for inner-cut fillet 
        rect_3 = draw.rectangle(p.ext_len, cpw_width) # outer part reaching cpw
        cir_1 = draw.Point(0,0).buffer(tee_width/2) # for the tee round edge 
        cir_2 = draw.Point(0,0).buffer(fillet) # for inner-cut fillet 

        # Merge and subtract
        tee = draw.union(
            rect_1, 
            draw.translate(rect_2, (tee_width+fillet)/2, 0),
            draw.translate(rect_3, (tee_width+p.ext_len)/2, 0),
            draw.translate(cir_1, 0, p.tee_length/2), 
            draw.translate(cir_1, 0,-p.tee_length/2), 
        )
        tee = draw.subtract(
            tee,
            draw.union(
                draw.translate(cir_2, tee_width/2+fillet, cpw_width/2+fillet),
                draw.translate(cir_2, tee_width/2+fillet,-cpw_width/2-fillet),
            )
        )

        # Rotate and then translate, relative to ref QComponent
        tee = draw.rotate(tee, Q.orientation+p.rel_ori, origin=(0,0))
        tee = draw.translate(
            tee, 
            Q.pos_x + p.shift_x, 
            Q.pos_y + p.shift_y
        )
        
        self.add_qgeometry(
            'poly', 
            dict(tee = tee), 
            chip = p.chip, 
            layer = p.layer, 
            subtract = subtract)
