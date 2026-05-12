import re

with open("tests/program/v2/ir/test_ir_passes_increg_merge.py", "r") as f:
    content = f.read()

# Fix the duplicate src="op" caused by fuzzy match replace
content = re.sub(r'src="op",\s*src="op", op=AluExpr', r'src="op", op=AluExpr', content)
content = re.sub(r'src="op",\s*op=AluExpr\(Register\("r2"\), "-", Literal\("#1"\)\)', r'src="op", op=AluExpr(Register("r2"), AluOp.SUB, ImmValue(1, prefix="#"))', content)
content = re.sub(r'Literal\("#([^"]+)"\)', r'ImmValue(\1, prefix="#")', content)
content = re.sub(r'op=AluExpr\(([^,]+), "\+",', r'op=AluExpr(\1, AluOp.ADD,', content)
content = re.sub(r'op=AluExpr\(([^,]+), "-",', r'op=AluExpr(\1, AluOp.SUB,', content)
content = re.sub(r'\.lit\.value == "#(\d+)"', r'.lit.value == \1', content)

with open("tests/program/v2/ir/test_ir_passes_increg_merge.py", "w") as f:
    f.write(content)
