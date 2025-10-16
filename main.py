import numpy as np
from typing import List

"""
This is a minimal implementation of an arbitrary size linear neural network, in numpy.

It is modeled heavily off of pytorch. Just a fun exercise to me to practice with!
It operates the same way as numpy, by defining tensor objects that have custom methods for arithmetic ops.
"""


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data: np.ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad: None | np.ndarray = None  # stores the gradient of the tensor.
        self._backward = (
            lambda: None
        )  # overwridden by arithmetic operations... basically computes the gradient
        self._previous_tensors: tuple = ()  # this is a list of tensors that were used to compute this tensor.
        self._op: str = "None"  # stores the operation that was used to compute this tensor for debugging.
        self._is_leaf: bool = True

    def __str__(self):
        return f"Tensor with data: \n {self.data} With requires_grad: {self.requires_grad} \n With grad: {self.grad}. Produced with op: {self._op}"

    def _get_topo_ordering(self) -> List["Tensor"]:
        """
        Runs a DFS to get a topo ordering of nodes, starting from this Tensor.
        """

        # topological order
        topo, visited = [], set()

        # this just explores the dependency tree
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    dfs(p)
                topo.append(v)

        dfs(self)  # build the topo ordering starting from this node...

        return topo

    def backwards(self, grad=None) -> None:
        """
        High level idea here:
        -> A topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.
        -> We need to propagate gradients from a loss term (end of the chain of computation) backwards towards the leafs.
        -> To do this, we run a DFS to get a topo ordering. Then we compute gradients in reverse topo ordering (think from the root down to the leafs)
        -> The way the gradient is computed of course depends on the operation used to produce the closer-to-root tensor.
        -> So to account for this we define _backward methods that depend on the type of operation used!
        """

        self.grad = grad if self.grad is None else self.grad + grad

        # topological order
        topo = self._get_topo_ordering()
        rev_topo = reversed(topo)

        for v in rev_topo:
            v._backward()  # compute the gradients from root -> leaves...

    ###################################
    # BEGIN TENSOR ARITHMETIC METHODS #
    ###################################

    ## NOTE: supported is Addition, Subtraction...

    def __add__(self, other: "Tensor") -> "Tensor":
        """
        uses this matrix derivative rule.

        Let:
            Z = X+Y
        Then:
            dZ/dX = I
            dZ/dY = I

        Thus, if we have dZ/dL, and we want dX/dL, dX/dL = dZ/dL.
        """
        if not isinstance(other, "Tensor"):
            raise TypeError(f"Tensors can only be added to other tensors")
        if self.data.shape != other.data.shape:
            raise NotImplementedError(
                "add: broadcasting not supported. Both tensors must be same shape."
            )

        new_tensor = Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        new_tensor._is_leaf = False
        new_tensor._previous_tensors = (self, other)
        new_tensor._op = "add"

        # we have now made a new tensor but we need to update the backwards method to propagate gradients appropriately to its children.

        if new_tensor.requires_grad:

            def _backward():
                if new_tensor.grad is None:
                    return
                if self.requires_grad:
                    self.grad = (
                        new_tensor.grad
                        if self.grad is None
                        else self.grad + new_tensor.grad
                    )
                if other.requires_grad:
                    other.grad += (
                        new_tensor.grad
                        if other.grad is None
                        else other.grad + new_tensor.grad
                    )
                return

        new_tensor._backward = _backward

        return new_tensor


if __name__ == "__main__":
    x = Tensor(np.ones(shape=(1, 2)))

    print(x)
