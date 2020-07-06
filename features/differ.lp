#defined state/1.
#defined compare/2.
#defined const/1.
#defined qualValue/3.
#defined delta/4.

#program fast_differ.

differ(C1, C2) :- compare(C1, C2), cardinality(C1, N, S),
                    not cardinality(C2, N, S).

#program fast_differ_set(num).

differ(C1, C2) :- compare(C1, C2, num), cardinality(C1, N, S),
                    not cardinality(C2, N, S).

#program standard_differ.
differ(C1, C2) :-  
    compare(C1, C2), belong(X, C1, S), not belong(X, C2, S).

differ(C1, C2) :-  
    compare(C1, C2), not belong(X, C1, S), belong(X, C2, S).

#program standard_differ_set(num).
differ(C1, C2) :-  
    compare(C1, C2, num), belong(X, C1, S), not belong(X, C2, S).

differ(C1, C2) :-  
    compare(C1, C2, num), not belong(X, C1, S), belong(X, C2, S).
        
#program optimal_differ_start.
differ_card(C1, C2) :- 
    compare(C1, C2), cardinality(C1, N, S), not cardinality(C2, N, S).

equiv_card(C1, C2) :- not differ_card(C1, C2), compare(C1, C2).

#program optimal_differ_end.
differ(C1, C2) :- differ_card(C1, C2).
differ(C1, C2) :-   equiv_card(C1, C2),
                    belong(X, C1, S), not belong(X, C2, S).
        
differ(C1, C2) :-   equiv_card(C1, C2),
                    not belong(X, C1, S), belong(X, C2, S).

#program optimal_differ_set_start(num).
differ_card(C1, C2, num) :- 
    compare(C1, C2, num), cardinality(C1, N, S), not cardinality(C2, N, S).

equiv_card(C1, C2, num) :- not differ_card(C1, C2, num), compare(C1, C2, num).

#program optimal_differ_set_end(num).
differ(C1, C2) :- differ_card(C1, C2, num).
differ(C1, C2) :-   equiv_card(C1, C2, num),
                    belong(X, C1, S), not belong(X, C2, S).
        
differ(C1, C2) :-   equiv_card(C1, C2, num),
                    not belong(X, C1, S), belong(X, C2, S).

#program feature_differ.
differ(F, G) :- compare(F, G), qualValue(F, V, S), not qualValue(G, V, S).
differ(F, G) :- compare(F, G), delta(S, T, F, V), not delta(S, T, G, V).