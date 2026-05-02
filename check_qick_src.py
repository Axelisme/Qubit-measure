from qick.tprocv2_assembler import get_src_type
try:
    print(f"r0: {get_src_type('r0')}")
    print(f"#5: {get_src_type('#5')}")
    print(f"5: {get_src_type('5')}")
except Exception as e:
    print(f"Error: {e}")
