import os
import glob

js_files = glob.glob("*.js")

fn_dict = {}

for fname in js_files:
    with open(fname, "r") as f:
        code = f.read()
    fn_dict[fname[:-3]] = code

js_files = glob.glob("plot_specific/*.js")
for fname in js_files:
    with open(fname, "r") as f:
        code = f.read()
    fn_dict[fname[14:-3]] = code

# Write jsfunctions file
with open("../biocircuits/jsfunctions.py", "w") as f:
    f.write("jsfuns = {\n")
    for module, code in fn_dict.items():
        f.write(f'    "{module}": """\n')
        f.write(code)
        f.write('\n""",\n')
    f.write("}")


