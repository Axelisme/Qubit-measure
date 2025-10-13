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
"""File contains dictionary for Rectangle and the make()."""

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

class Cross(QComponent):
    """A cross.

    Inherits QComponent class.

    The class will add default_options class Dict to QComponent class before calling make.

    .. image::
        Cross.png

    .. meta::
        Cross

    Default Options:
        * width: '100um'
        * height: '100um'
        * trace_width: '5um'
        * orientation: 0
        * subtract: 'False'
        * helper: 'False'
    """
    component_metadata = Dict(short_name='Cross')
    """Component metadata"""

    default_options = Dict(width='100um', height='100um', trace_width='5um', orientation=0, subtract=False, helper=False)
    """Default connector options"""

    TOOLTIP = """A cross."""
    def make(self):
        """The make function implements the logic that creates the geoemtry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed information, such
        as layer, subtract, etc."""
        p = self.p  # p for parsed parameters. Access to the parsed options.

        # create the geometry
        rect1 = draw.rectangle(p.width, p.trace_width, p.pos_x, p.pos_y)
        rect2 = draw.rectangle(p.trace_width, p.height, p.pos_x, p.pos_y)
        cross = draw.union(rect1, rect2)
        cross = draw.rotate(cross, p.orientation)
        ##############################################
        # add qgeometry
        self.add_qgeometry('poly', {'cross': cross},
                           subtract=p.subtract,
                           helper=p.helper,
                           layer=p.layer,
                           chip=p.chip)

class Fillet_vertex(QComponent):
    """Tool shape for creating fillet objects.

    Inherits QComponent class.

    The class will add default_options class Dict to QComponent class before calling make.

    .. image::
        Fillet_vertex.png

    .. meta::
        Fillet_vertex

    Default Options:
        * fillet: '50um'
        * subtract: 'False'
        * helper: 'False'
    """
    default_options = Dict(fillet='50um',
                           subtract='False',
                           helper='False')
    """Default drawing options"""

    TOOLTIP = """A single configurable square with fillet"""

    def make(self):
        """The make function implements the logic that creates the geoemtry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed information, such
        as layer, subtract, etc."""
        p = self.p  # p for parsed parameters. Access to the parsed options.

        # create the geometry

        square = draw.rectangle(p.fillet, p.fillet, p.pos_x+p.fillet/2, p.pos_y+p.fillet/2)
        circle = draw.Point(p.pos_x, p.pos_y).buffer(p.fillet)
        fil_ver = draw.subtract(square, circle)
        fil_ver = draw.rotate(fil_ver, p.orientation)
        
        ##############################################
        # add qgeometry
        self.add_qgeometry('poly', {'fillet_vertex': fil_ver},
                           subtract=p.subtract,
                           helper=p.helper,
                           layer=p.layer,
                           chip=p.chip)

class Rect_fillet(QComponent):
    """A single configurable square with fillet at vertices.

    Inherits QComponent class.

    The class will add default_options class Dict to QComponent class before calling make.

    .. image::
        Rectangle_with_fillet.png

    .. meta::
        Rectangle_with_fillet

    Default Options:
        * width: '500um'
        * height: '300um'
        * fillet: '50um'
        * subtract: 'False'
        * helper: 'False'
    """

    default_options = Dict(width='500um',
                           height='300um',
                           fillet='50um',
                           subtract='False',
                           helper='False')
    """Default drawing options"""

    TOOLTIP = """A single configurable square with fillet"""

    def make(self):
        """The make function implements the logic that creates the geoemtry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed information, such
        as layer, subtract, etc."""
        p = self.p  # p for parsed parameters. Access to the parsed options.

        # create the geometry
        rect1 = draw.rectangle(p.width-2*p.fillet, p.height, p.pos_x, p.pos_y)
        rect2 = draw.rectangle(p.width, p.height-2*p.fillet, p.pos_x, p.pos_y)
        cir = draw.Point(p.pos_x, p.pos_y).buffer(p.fillet)
        x,y = +p.width/2-p.fillet, +p.height/2-p.fillet
        rect_fil = draw.union(rect1, rect2, 
                              draw.translate(cir, x, y),
                              draw.translate(cir,-x, y),
                              draw.translate(cir, x,-y),
                              draw.translate(cir,-x,-y),
                              )
        ##############################################
        # add qgeometry
        self.add_qgeometry('poly', {'rectangle': rect_fil},
                           subtract=p.subtract,
                           helper=p.helper,
                           layer=p.layer,
                           chip=p.chip)
