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
"""Qubit with pads of fillet vertices and tappered arms.

.. code-block::
     ________________________________
    (                                )
    (                                )
    |        __________________      |
    |       (                  )     |
    |       (__________________)     |
    |                 |              |
    |                 x              |
    |        _________|________      |
    |       (                  )     |
    |       (__________________)     |
    |                                |
    (                                )
    (________________________________)
"""

import warnings

try:
    from qiskit_metal import Dict, draw  # type: ignore[reportMissingImports]
    from qiskit_metal.qlibrary.core import (  # type: ignore[reportMissingImports]
        BaseQubit,
    )
except ImportError:
    print("qiskit_metal is not installed")
    raise


class Fillet_Qubit(BaseQubit):
    """The base `Fillet_Qubit` class.

    Inherits `BaseQubit` class.

    Create a pocket for a ground plane,
    with two pads with fillet vertices and tappered arms connected by Josephson junctions.

    Connector lines can be added using the `connection_pads`
    dictionary. Each connector pad has a name and a list of default
    properties.

    Sketch:
        Below is a sketch of the qubit
        ::

                 +1                            +1
                ________________________________
            -1  |______ ____           __________|   +1     Y
                |      |____|         |____|     |          ^
                |        __________________      |          |
                |       |     island       |     |          |----->  X
                |       |__________________|     |
                |                 |              |
                |  pocket         x              |
                |        _________|________      |
                |       |                  |     |
                |       |__________________|     |
                |        ______                  |
                |_______|______|                 |
            -1  |________________________________|   +1

                 -1                            -1

    .. image::
        Fillet_Qubit.png

    .. meta::
        Fillet Qubit

    BaseQubit Default Options:
        * connection_pads: Empty Dict -- The dictionary which contains all active connection lines for the qubit.
        * _default_connection_pads: Empty Dict -- The default values for the (if any) connection lines of the qubit.

    Default Options:
        * arm_width: '8um' --
        * arm_length: '50um' --
        * arm_fillet: '25um' --
        * pad_gap: '30um' -- The distance between the two charge islands, which is also the resulting 'length' of the pseudo junction
        * pad_width: '150um' -- The width (x-axis) of the charge island pads
        * pad_height: '150um' -- The size (y-axis) of the charge island pads
        * pad_fillet: '20um' --
        * pocket_width: '400um' -- Size of the pocket (cut out in ground) along x-axis
        * pocket_height: '600um' -- Size of the pocket (cut out in ground) along y-axis
        * pocket_fillet: '100um' --
        * _default_connection_pads: Dict
            * pad_gap: '15um' -- Space between the connector pad and the charge island it is nearest to
            * pad_width: '125um' -- Width (x-axis) of the connector pad
            * pad_height: '30um' -- Height (y-axis) of the connector pad
            * pad_cpw_shift: '5um' -- Shift the connector pad cpw line by this much away from qubit
            * pad_cpw_extent: '25um' -- Shift the connector pad cpw line by this much away from qubit
            * cpw_width: 'cpw_width' -- Center trace width of the CPW line
            * cpw_gap: 'cpw_gap' -- Dielectric gap width of the CPW line
            * cpw_extend: '100um' -- Depth the connector line extense into ground (past the pocket edge)
            * pocket_extent: '5um' -- How deep into the pocket should we penetrate with the cpw connector (into the fround plane)
            * pocket_rise: '65um' -- How far up or downrelative to the center of the transmon should we elevate the cpw connection point on the ground plane
            * loc_W: '+1' -- Width location  only +-1
            * loc_H: '+1' -- Height location only +-1
    """

    default_options = Dict(
        arm_width="8um",
        arm_length="50um",
        arm_fillet="25um",
        pad_gap="30um",
        pad_width="150um",
        pad_height="150um",
        pad_fillet="20um",
        pocket_width="400um",
        pocket_height="600um",
        pocket_fillet="100um",
        # 90 has dipole aligned along the +X axis,
        # while 0 has dipole aligned along the +Y axis
        _default_connection_pads=Dict(
            pad_gap="15um",
            pad_width="125um",
            pad_height="30um",
            pad_cpw_shift="5um",
            pad_cpw_extent="25um",
            cpw_width="cpw_width",
            cpw_gap="cpw_gap",
            # : cpw_extend: how far into the ground to extend the CPW line from the coupling pads
            cpw_extend="100um",
            pocket_extent="5um",
            pocket_rise="65um",
            loc_W="+1",  # width location  only +-1
            loc_H="+1",  # height location only +-1
        ),
    )
    """Default drawing options"""

    component_metadata = Dict(
        short_name="Pocket",
        _qgeometry_table_path="True",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
    )
    """Component metadata"""

    TOOLTIP = """The user component `Fillet_Qubit` class."""

    def make(self):
        """Define the way the options are turned into QGeometry.

        The make function implements the logic that creates the geoemtry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed
        information, such as layer, subtract, etc.
        """
        self.make_pocket()

    def make_pocket(self):
        """Makes standard transmon in a pocket."""
        # self.p allows us to directly access parsed values (string -> numbers) form the user option
        p = self.p
        # extract chip name
        chip = p.chip
        # main pad
        max_pad_fillet = min(
            (p.pad_width - 2 * p.arm_width) / 2,
            (p.pad_height - p.arm_width) / 2,
        )
        if p.pad_fillet > max_pad_fillet:
            warnings.warn(
                f"pad_fillet is larger than the maximum fillet size. Setting it to {max_pad_fillet}"
            )
            p.pad_fillet = max_pad_fillet
        rect1 = draw.rectangle(p.pad_width, p.pad_height - 2 * p.pad_fillet)
        rect2 = draw.rectangle(p.pad_width - 2 * p.pad_fillet, p.pad_height)
        cir1 = draw.Point(0, 0).buffer(p.pad_fillet)
        # arm and tapper
        max_arm_fillet = min(
            (p.pad_width - 2 * p.pad_fillet - p.arm_width) / 2,
            p.arm_length,
        )
        if p.arm_fillet > max_arm_fillet:
            warnings.warn(
                f"arm_fillet is larger than the maximum fillet size. Setting it to {max_arm_fillet}"
            )
            p.arm_fillet = max_arm_fillet
        arm_fillet = p.arm_fillet

        rect3 = draw.rectangle(p.arm_width + 2 * arm_fillet, arm_fillet)
        cir2 = draw.Point(0, 0).buffer(arm_fillet)
        rect4 = draw.rectangle(p.arm_width, p.arm_length - arm_fillet)
        # Union and create both pads
        x, y = p.pad_width / 2 - p.pad_fillet, p.pad_height / 2 - p.pad_fillet
        main_pad = draw.union(
            rect1,
            rect2,
            draw.translate(cir1, x, y),
            draw.translate(cir1, x, -y),
            draw.translate(cir1, -x, y),
            draw.translate(cir1, -x, -y),
        )
        x, y = (p.arm_width + 2 * arm_fillet) / 2, -arm_fillet / 2
        arm_pad_connect = draw.subtract(rect3, draw.translate(cir2, x, y))
        arm_pad_connect = draw.subtract(arm_pad_connect, draw.translate(cir2, -x, y))
        pad_top = draw.union(
            draw.translate(main_pad, 0, p.arm_length + (p.pad_height + p.pad_gap) / 2),
            draw.translate(
                arm_pad_connect,
                0,
                p.arm_length + (p.pad_gap - arm_fillet) / 2,
            ),
            draw.translate(rect4, 0, (p.arm_length - arm_fillet + p.pad_gap) / 2),
        )
        pad_bot = draw.rotate(pad_top, 180, origin=(0, 0))
        # JJ
        rect_jj = draw.LineString([(0, -p.pad_gap / 2), (0, +p.pad_gap / 2)])
        # pocket
        rect4 = draw.rectangle(p.pocket_width, p.pocket_height - 2 * p.pocket_fillet)
        rect5 = draw.rectangle(p.pocket_width - 2 * p.pocket_fillet, p.pocket_height)
        cir3 = draw.Point(0, 0).buffer(p.pocket_fillet)
        x, y = (
            p.pocket_width / 2 - p.pocket_fillet,
            p.pocket_height / 2 - p.pocket_fillet,
        )
        rect_pk = draw.union(
            rect4,
            rect5,
            draw.translate(cir3, x, y),
            draw.translate(cir3, x, -y),
            draw.translate(cir3, -x, y),
            draw.translate(cir3, -x, -y),
        )
        # Rotate and translate all qgeometry as needed.
        polys = [rect_jj, pad_top, pad_bot, rect_pk]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)
        [rect_jj, pad_top, pad_bot, rect_pk] = polys
        # Use the geometry to create Metal qgeometry
        self.add_qgeometry("poly", dict(pad_top=pad_top, pad_bot=pad_bot), chip=chip)
        self.add_qgeometry("poly", dict(rect_pk=rect_pk), subtract=True, chip=chip)
        self.add_qgeometry(
            "junction",
            dict(rect_jj=rect_jj),
            width=p.arm_width,
            chip=chip,
        )
