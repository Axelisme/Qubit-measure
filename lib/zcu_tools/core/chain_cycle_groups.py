"""Read groups of the (D R R)^8 cycle behind its 65 stored numbers.

A cycle has 16 reads in time order. Read 2k follows drive D_k and read 2k+1
follows a read. chain_cycle_sums sums I², IQ, Q² per same-read group and IQ
products per pair group; the cycle model of analysis uses the same groups.
These two functions are the single definition of that grouping.
"""


def same_read_group(read: int) -> int:
    """Return the same-read group (0, 1 or 2) of a read index in 0..15.

    Group 0 is read 0. Group 1 is the other reads that follow a drive, reads
    2, 4, ..., 14. Group 2 is the reads that follow a read, reads 1, 3, ..., 15.
    An index outside 0..15 is outside this contract and is not checked.
    """
    if read == 0:
        return 0
    return 1 if read % 2 == 0 else 2


def pair_group(left: int, right: int) -> int:
    """Return the pair group (0..5) of reads left < right, both in 0..15.

    With d the number of drives between the reads and gap = right - left, the
    groups are 0: (d=0, gap=1); 1: (d=1, gap=1); 2: (d=1, gap=2..3); 3: (d=2,
    gap=2..3); 4: (d=2, gap>=4); 5: (d>=3), holding 8, 7, 21, 6, 18 and 60 pairs.
    Indices outside this contract are not checked.
    """
    drives = right // 2 - left // 2
    gap = right - left
    if drives == 0:
        return 0
    if drives == 1:
        return 1 if gap == 1 else 2
    if drives == 2:
        return 3 if gap <= 3 else 4
    return 5
