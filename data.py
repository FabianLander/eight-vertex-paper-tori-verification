"""The data the paper prints, transcribed once and read by every check.

  Q_SQUARE, P_HEX_ROWS         Table 1, the two folded bases
  FACES_SQUARE, FACES_HEX      the two face lists, in the printed order
  SIGNS_SQUARE, SIGNS_HEX      the two orientation sign strings
  ZETA_SQUARE, ZETA_HEX        the lift directions zeta and zeta'
  LOOPS_SQUARE, LOOPS_HEX      the two marking loops of each triangulation
  FREE_SQUARE, FREE_HEX        the nine free coordinates of each Jacobian

EDGES_SQUARE and EDGES_HEX are derived, not printed: the paper reads the
edges off the face cycles, and the combinatorics check compares its own
derived edge set against these.
"""
from fractions import Fraction as Fr

# ------------------------------------------------- Table 1, square column
Q_SQUARE = {
    0: (Fr(0), Fr(0)),
    1: (Fr(73, 60), Fr(11, 10)),
    2: (Fr(1, 2), Fr(0)),
    3: (Fr(-15977, 50320), Fr(42471, 50320)),
    4: (Fr(1), Fr(0)),
    5: (Fr(35937, 25160), Fr(27357, 25160)),
    6: (Fr(0), Fr(33, 20)),
    7: (Fr(1), Fr(33, 20)),
}

# --------------------------------------------- Table 1, hexagonal column
# vertex: (common denominator, a_x, b_x, a_y, b_y), the vertex sitting at
# ((a_x + b_x sqrt3)/den, (a_y + b_y sqrt3)/den).
P_HEX_ROWS = {
    0: (1, 0, 0, 0, 0),
    1: (8073677900, 896842303, 729553300, 1029317275, 568256878),
    2: (47100, -2625, 7310, 8925, -24854),
    3: (10525, 3789, 0, 595, -4713),
    4: (1, 1, 0, 0, 0),
    5: (200, 64, 25, 75, -136),
    6: (200, -36, 25, 75, -36),
    7: (109900, 279568, -117825, 264900, -143557),
}


def hex_base(field):
    """The hexagonal column of Table 1 over any representation of
    a + b*sqrt3 built from (a, b)."""
    out = {}
    for v, (den, ax, bx, ay, by) in P_HEX_ROWS.items():
        out[v] = (field(Fr(ax, den), Fr(bx, den)),
                  field(Fr(ay, den), Fr(by, den)))
    return out


# ------------------------------------------------------------ the face lists
FACES_SQUARE = [(3, 5, 6), (3, 2, 5), (3, 6, 4), (3, 0, 2), (3, 4, 1), (3, 1, 0),
                (5, 0, 6), (5, 2, 4), (5, 4, 7), (5, 7, 0), (6, 7, 4), (6, 0, 1),
                (6, 1, 7), (2, 1, 4), (2, 0, 7), (2, 7, 1)]
FACES_HEX = [(0, 1, 2), (0, 3, 1), (0, 2, 4), (0, 4, 3), (1, 5, 2), (1, 3, 6),
             (1, 4, 5), (1, 7, 4), (1, 6, 7), (2, 6, 4), (2, 5, 7), (2, 7, 6),
             (3, 4, 7), (3, 5, 6), (3, 7, 5), (4, 6, 5)]

# the 24 edges of each list, read off the face cycles above
EDGES_SQUARE = [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3),
                (1, 4), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4),
                (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
EDGES_HEX = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 5),
             (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5),
             (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]

# --------------------------------------------------- the printed sign strings
SIGNS_SQUARE = '++-++--+-+-++---'
SIGNS_HEX = '-++-----+-++++-+'

# ------------------------------------------------------- the lift directions
ZETA_SQUARE = (Fr(3, 13), Fr(-3, 8), Fr(-3, 7), Fr(12, 13),
               Fr(1), Fr(-1), Fr(1), Fr(0))
ZETA_HEX = (Fr(1, 2), Fr(-1), Fr(6, 17), Fr(-7, 34),
            Fr(-33, 47), Fr(-1), Fr(19, 41), Fr(1))

# ------------------------------------------------------------- the markings
LOOPS_SQUARE = [[4, 2, 3, 4], [5, 2, 0, 5]]
LOOPS_HEX = [[1, 4, 0, 1], [1, 5, 6, 1]]

# ------------------------------- the free coordinates of the two Jacobians
FREE_SQUARE = ('x0', 'y0', 'x1', 'y1', 'y2', 'x3', 'x4', 'y4', 'x5')
FREE_HEX = ('x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4')
