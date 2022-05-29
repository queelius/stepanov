Suppose we have a set G and two operations,
    `op : (G,G) -> G`
and
    `inv : G -> G`.

Let `a, b, c, ...` denote any element of `G`.
The tuple `(G,op,id,inv)` is a group if the
following set of axioms are satisfied.

The first axiom is associativity,
    `op(op(a,b),c) = op(a,op(b,c))`.

As a result, we may unambiguously write
`op(a,b,c)` or more generally `op(a,b,...)`.
Additionally, from a computational
perspective, associativity permits
regrouping for, say, efficiency, without
changing the result, e.g.,
    `op(a,b,c,d,e,f,g,h)`
may be decomposed into
    `op(op(op(a,b),op(c,d)),
         op(op(e,f),op(g,h)))`,
which is a tree that permits simuntaneously computating sibling
sub-expressions.

The second axiom is unique identity. For
all `a` in `G`, there exists a unique
element `e` in `G` such that
    `op(e,a) = op(a,e) = a`.
We denote e the identity element `e` under
`op : (G,G) -> G` such that
`op(e,x) = op(x,e) = x`.

The third axiom is invertibility,
    `op(a,inv(a)) = op(inv(a),a) = id()`.

We use `G` as a shorthand for the group
`(G,op,id,inv)`.

Note: `op` may be non-commutative, i.e.,
      `op(a,b) = op(b,a)` is not necessary.
