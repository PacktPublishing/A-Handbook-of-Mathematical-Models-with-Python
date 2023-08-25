
from pulp import *

#value per weight
v = {'Sleeping bag': 4.17,
'Pillow': 5.13,
'Torch': 10.0,
'First Aid Kit': 8.0,
'Hand sanitiser': 2.0}

#weight
w = {'Sleeping bag':1.2,
'Pillow':0.39,
'Torch':0.5,
'First Aid Kit':0.5,
'Hand sanitiser':0.5}

limit = 2.9
items = list(sorted(v.keys()))

# Model
m = LpProblem("Knapsack Problem", LpMaximize)

# Variables
x = LpVariable.dicts('x', items, lowBound = 0, upBound = 1, 
                     cat = LpInteger)
# Objective
m += sum(v[i]*x[i] for i in items)

# Constraint
m += sum(w[i]*x[i] for i in items) <= limit

# Optimize
m.solve()

#print("Status = %s" % LpStatus[m.status])

for i in items:
    print("%s = %f" % (x[i].name, x[i].varValue))

#print("Objective = %f" % value(m.objective))
