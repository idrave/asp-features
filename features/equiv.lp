#program fast_equiv.

differ(C1, C2) :- compare(C1, C2), state(S), cardinality(C1, N, S),
                    not cardinality(C2, N, S).

#program standard_equiv.
differ(C1, C2) :-   compare(C1, C2), const(X), state(S),
                    belong(X, C1, S), not belong(X, C2, S).
        
differ(C1, C2) :-   compare(C1, C2), const(X), state(S),
                    not belong(X, C1, S), belong(X, C2, S).

#program optimal_equiv.
differ_card(C1, C2) :- compare(C1, C2), state(S), cardinality(conc(C1), N, S),
                    not cardinality(conc(C2), N, S).
equiv_card(C1, C2) :- not differ_card(C1, C2), compare(C1, C2).

differ(C1, C2) :- differ_card(C1, C2).
differ(C1, C2) :-   equiv_card(C1, C2), const(X), state(S),
                    belong(X, C1, S), not belong(X, C2, S).
        
differ(C1, C2) :-   equiv_card(C1, C2), const(X), state(S),
                    not belong(X, C1, S), belong(X, C2, S).