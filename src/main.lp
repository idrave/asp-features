#include "concept.lp".
#include "prune.lp".
#include "features.lp".

#script(python).
import clingo
def main(prg):

    prg.ground( [("base", [])] )
    k = prg.get_const("k")
    k = k.number
    print("Expand until: {}".format(k))
    for i in range(1, k+1):
        prg.ground( [("step", [i]), ("reduce", [i])] ) 
        prg.solve()
        prg.cleanup()
    prg.ground([("features", [])])
    prg.solve()
#end.

#show conc/2.
%#show bool/1.
%#show num/1.