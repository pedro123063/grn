import cProfile as cpr
import re
cpr.run('re.compile("foo|bar")')

import matrix as ma
k=500

m1 = ma.criar(k)
m2= ma.criar(k)

m3= ma.multiplicar(m1,m2)