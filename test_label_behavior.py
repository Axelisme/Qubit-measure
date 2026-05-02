from zcu_tools.program.v2.ir.labels import Label
from copy import deepcopy

def test_label_clone():
    Label.reset()
    l1 = Label.make_new("test")
    print(f"Original: {l1}")
    
    l2 = deepcopy(l1)
    print(f"Clone 1: {l2}")
    
    l3 = deepcopy(l1)
    print(f"Clone 2: {l3}")
    
    # Check shared reference within a subtree
    memo = {}
    l_orig = Label.make_new("shared")
    subtree = [l_orig, l_orig]
    cloned_subtree = deepcopy(subtree, memo)
    print(f"Shared Original: {subtree[0]} == {subtree[1]}? {subtree[0] is subtree[1]}")
    print(f"Shared Clone: {cloned_subtree[0]} == {cloned_subtree[1]}? {cloned_subtree[0] is cloned_subtree[1]}")
    print(f"Shared Clone Names: {cloned_subtree[0]}, {cloned_subtree[1]}")

if __name__ == "__main__":
    test_label_clone()
