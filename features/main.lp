#include "concept.lp".
#include "prune.lp".
#include "features.lp".
#include "equiv.lp".
#include "sat.lp".

#script(python).
def save_file(m):
    print("ola")
    with open("/home/ivan/Documents/ai/features/tmp/feat.out", 'w') as file:
        file.write(str(m))

import clingo
def main(prg):

    prg.ground( [("base", [])] )
    k = prg.get_const("k")
    k = k.number
    print("Expand until: {}".format(k))
    for i in range(1, k+1):
        prg.ground( [("step", [i]),("fast_equiv", []),("reduce", [i])] ) 
        with prg.solve(yield_=True) as models:
            if i == k:
                with open("concepts{}.lp".format(k),'w') as file:
                    for model in models: file.write(str(model))
        prg.cleanup()
    prg.ground([("features", [])])
    prg.solve()
    print("###### STARTING SAT ######")
    prg.ground([("sat", [])])
    prg.solve()
#end.

%#show selected/1.
%#show feature/1.
%#show cardinality/3.
#show conc/2.
%#show num/1.
%#show bool/1.
%#show num/1.