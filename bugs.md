## Bugs found in refactor

1. Random scaling spread is always ```exp(unif([0, 1]))``` instead of ```exp(unif(-L, L))```.
2. Global parameter RNN uses wrong state information.