"""
A Distributor is an object that contains WeightAcceptors. It has a distribute method which, when called,
distributes a given amount of weight to the WeightAcceptors. It promises to take every measure in its power
to ensure that the given is always accepted by the weightAcceptors; however, it is not always successful,
and when it is not successful, this is normal. It does not give warnings or errors when it fails to distribute
the prescribed amount of weight.

A WeightAcceptor is an object that has an accept_weight method that takes a suggested amount of weight
and returns the quantity of weight that was actually accepted. The amount of weight accepted must have the
same sign as the suggested amount of weight. For example, if it is offered a positive amount of weight,
it must not accept a negative amount of weight. Furthermore, it must always state the maximum amount of weight
it will ever be able to take if it were offered an infinite amount of weight, including if the cumulative sum
of the amount of weight it took over time were infinite. (These two numbers must be the same.)

A Distributor can be a WeightAcceptor.

A GeometricDistributor is a special kind of Distributor. It maintains the invariant that the total amount of weight
offered to its (i+1)th weight acceptor is differs from the total amount of weight offered to its ith weight
acceptor by a fixed constant factor that is between zero and one.
"""
