pred(on).
pred(clear).

arity(on, 2).
arity(clear, 1).

const(a).
const(b).
const(c).
const(floor).

state(state0).
hold(state0, clear, a).
hold(state0, on, (a, b)).
hold(state0, on, (b, c)).
hold(state0, on, (c, floor)).