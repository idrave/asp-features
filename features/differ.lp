#program fast_differ(depth, round).

differ(C1, C2) :- compare(C1, C2, depth, round), state(S), cardinality(C1, N, S),
                    not cardinality(C2, N, S).

#program standard_differ(depth, round).
current(round).
differ(C1, C2) :-  
    compare(C1, C2, depth, round), const(X), state(S),
    belong(X, C1, S), not belong(X, C2, S).

differ(C1, C2) :-  
    compare(C1, C2, depth, round), const(X), state(S),    
    not belong(X, C1, S), belong(X, C2, S).
        

#program optimal_differ_start(depth, round).
differ_card(C1, C2, depth, round) :- 
    compare(C1, C2, depth, round), state(S),
    cardinality(conc(C1), N, S), not cardinality(conc(C2), N, S).

equiv_card(C1, C2, depth, round) :- not differ_card(C1, C2, depth, round), compare(C1, C2, depth, round).

#program optimal_differ_end(depth, round).
differ(C1, C2) :- differ_card(C1, C2, depth, round).
differ(C1, C2) :-   equiv_card(C1, C2, depth, round), const(X), state(S),
                    belong(X, C1, S), not belong(X, C2, S).
        
differ(C1, C2) :-   equiv_card(C1, C2, depth, round), const(X), state(S),
                    not belong(X, C1, S), belong(X, C2, S).