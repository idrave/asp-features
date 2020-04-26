pred(boss).
pred(human).

arity(human, 1).
arity(boss, 2).

const(alice).
const(charlie).
const(bob).

state(st0).

hold(st0, human, alice).
hold(st0, human, bob).
hold(st0, human, charlie).
hold(st0, boss, (alice, alice)).
hold(st0, boss, (alice, bob)).
hold(st0, boss, (alice, charlie)).
