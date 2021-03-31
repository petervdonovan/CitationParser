"""
An IndependentUnionWeightAcceptor is a WeightAcceptor whose weight behaves "like the union of
independent events." Unfortunately, it is a little more complicated than a LinearStubbornWeightAcceptor,
which takes all the weight that it is given and then refuses to take more weight after it is told to be
stubborn. It is a good idea to make GeometricDistributors be LinearStubbornWeightAcceptors; in fact,
LinearStubbornWeightAcceptors are the nicest WeightAcceptors possible, and it is recommended that they
be used unless other methods are truly necessary.

Like all WeightAcceptors, an IndependentUnionWeightAcceptor really ought to state the derivative
of its weight with respect to the weight that it is offered. It's simply a matter of basic courtesy.
Good Distributors should be able to use WeightAcceptor derivatives to compute the derivative of weight
accepted versus some parameter, which can be modified with the use of Newton's method or similar.

WeightAcceptors should have a reset method as well.
"""
