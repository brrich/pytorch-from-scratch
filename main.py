import numpy as np
from typing import List

"""
This is a minimal implementation of an arbitrary size linear neural network, in numpy.

It is modeled heavily off of pytorch. Just a fun exercise to me to practice autograd.

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
                for p in v._previous_tensors:
                    dfs(p)
                topo.append(v)

        dfs(self)  # build the topo ordering starting from this node...

        return topo

    @staticmethod
    def sum_to_shape(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """
        reduces a broadcasted gradient `g` back to an operand's original `shape`.

        in general this useful so we can have things like biases. also just works nicely with numpy broadcasting

        Broadcasting (NumPy/PyTorch) virtually "expands" smaller operands to the
        output shape. In reverse-mode autodiff, the transpose of that expand is a
        SUM over the expanded (broadcasted) axes. This routine implements exactly
        that reduction.

        Notes
        - Shapes are right-aligned (NumPy broadcasting rules).
        - If `shape` has fewer dims than `g`, we sum away the extra leading axes.
        - For any aligned axis where `shape[i] == 1` and `g.shape[i] > 1`,
          we sum over that axis with `keepdims=True`.

        Examples
        --------
        >>> # bias grad in a linear layer
        >>> # Out = X + b, X.shape = (B, D), b.shape = (D,)
        >>> Tensor.sum_to_shape(np.ones((4, 3)), (3,)).shape
        (3,)
        >>> # scalar add: X + c, c.shape = ()
        >>> Tensor.sum_to_shape(np.ones((2, 2)), ()).shape
        ()
        """
        # remove leading axes present in g but not in shape
        while g.ndim > len(shape):
            g = g.sum(axis=0)

        # collapse broadcasted axes (where operand had size 1)
        for i, (gd, sd) in enumerate(zip(g.shape, shape)):
            if sd == 1 and gd != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

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
        Elementwise addition with NumPy-tyle broadcasting.

        uses this matrix derivative rule for matrix addition

        Let:
            Z = X+Y
        Then:
            dZ/dX = I
            dZ/dY = I

        Thus, if we have dL/dZ, and we want dL/dX, dL/dX = dL/dZ * dZ/dX = dL/dZ.
        """
        # autoconvert to tensor if not tensor
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other, dtype=self.data.dtype))

        # NOTE: this relies on potential numpy broadcasting, hence the need for sum_to_shape.
        out = Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._is_leaf = False
        out._previous_tensors = (self, other)
        out._op = "add"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                if self.requires_grad:
                    g_self = Tensor.sum_to_shape(G, self.data.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = Tensor.sum_to_shape(G, other.data.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other

            out._backward = _backward

        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        Elementwise multiplication with NumPy-style broadcasting.

        uses this matrix derivative rule for elementwise multiplication

        Let:
            Z = X * Y
        Then:
            dZ/dX = Y
            dZ/dY = X

        Thus, if we have dL/dZ, and we want dL/dX,
            dL/dX = dL/dZ * dZ/dX = dL/dZ * Y
        and similarly for Y.
        """
        # autoconvert to tensor if not tensor
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other, dtype=self.data.dtype))

        # NOTE: this relies on potential numpy broadcasting, hence the need for sum_to_shape.
        out = Tensor(
            data=self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._is_leaf = False
        out._previous_tensors = (self, other)
        out._op = "mul"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                if self.requires_grad:
                    g_self = Tensor.sum_to_shape(G * other.data, self.data.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = Tensor.sum_to_shape(G * self.data, other.data.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other

            out._backward = _backward

        return out

    def __pow__(self, power: "Tensor | float | int") -> "Tensor":
        """
        Elementwise power with NumPy-style broadcasting.

        uses these derivative rules

        Let:
            Z = X ** P
        Then:
            dZ/dX = P * X**(P-1)
            dZ/dP = (X**P) * ln(X)        # when P is a Tensor

        Thus, if we have dL/dZ:
            dL/dX = dL/dZ * (P * X**(P-1))
            dL/dP = dL/dZ * (X**P * ln(X))
        (the dL/dP term only applies when the exponent is a Tensor with requires_grad)

        Notes
        - For X <= 0, ln(X) is undefined; NumPy will yield inf/NaN accordingly.
        """
        # autoconvert to tensor if not tensor
        if not isinstance(power, Tensor):
            power = Tensor(np.array(power, dtype=self.data.dtype))

        out = Tensor(
            data=self.data**power.data,
            requires_grad=self.requires_grad or power.requires_grad,
        )
        out._is_leaf = False
        out._previous_tensors = (self, power)
        out._op = "pow"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return

                if self.requires_grad:
                    # dZ/dX = P * X**(P-1)
                    base_term = power.data * np.power(self.data, power.data - 1)
                    g_self = Tensor.sum_to_shape(G * base_term, self.data.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self

                if power.requires_grad:
                    # dZ/dP = (X**P) * ln(X) = out * ln(X)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        exp_term = out.data * np.log(self.data)
                    g_pow = Tensor.sum_to_shape(G * exp_term, power.data.shape)
                    power.grad = g_pow if power.grad is None else power.grad + g_pow

            out._backward = _backward

        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        matrix multiply (2D only): (m x n) @ (n x p) -> (m x p)

        if we want A@B

            dL/dA = dL/dZ @ B^T
            dL/dB = A^T @ dL/dZ

        """
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other, dtype=self.data.dtype))

        A, B = self.data, other.data
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("matmul expects 2D tensors")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch: {A.shape} @ {B.shape}")

        out = Tensor(A @ B, requires_grad=self.requires_grad or other.requires_grad)
        out._is_leaf = False
        out._previous_tensors = (self, other)
        out._op = "matmul"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                if self.requires_grad:
                    gA = G @ B.T
                    self.grad = gA if self.grad is None else self.grad + gA
                if other.requires_grad:
                    gB = A.T @ G
                    other.grad = gB if other.grad is None else other.grad + gB

            out._backward = _backward

        return out

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, self.requires_grad)
        out._is_leaf = False
        out._previous_tensors = (self,)
        out._op = "neg"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                g = -G
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    # these are also useful...

    def sum(self) -> "Tensor":
        """
        sum over all elements

        let:
            L = Σ_i X_i

        then the local derivative is:
            dL/dX = 1   (same shape as X)

        in backprop with upstream scalar grad G:
            d(loss)/dX = G * 1  (G broadcasts)
        """
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out._is_leaf = False
        out._previous_tensors = (self,)
        out._op = "sum"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                if self.requires_grad:
                    g = np.ones_like(self.data) * G  # scalar G broadcasts
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        """
        mean over all elements

        let:
            N = X.size
            L = (1/N) * Σ_i X_i

        then the local derivative is:
            dL/dX = (1/N)   (same shape as X)

        in backprop with upstream scalar grad G:
            d(loss)/dX = (G/N)  (G broadcasts)
        """
        N = self.data.size
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        out._is_leaf = False
        out._previous_tensors = (self,)
        out._op = "mean"

        if out.requires_grad:

            def _backward():
                G = out.grad
                if G is None:
                    return
                if self.requires_grad:
                    g = (np.ones_like(self.data) / N) * G
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def __rpow__(self, other: "Tensor | float | int") -> "Tensor":
        return other.__pow__(self)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __radd__(self, other: "Tensor") -> "Tensor":
        return other.__add__(self)

    def __rsub__(self, other: "Tensor") -> "Tensor":
        return other.__sub__(self)

    def __rmul__(self, other: "Tensor") -> "Tensor":
        return self.__mul__(other)


def zero_grad(params: List[Tensor]) -> None:
    """resets gradients"""
    for p in params:
        p.grad = None


def sgd(params: List[Tensor], lr: float = 1e-2) -> None:
    """applies gradient descent."""
    for p in params:
        if p.grad is None:
            continue
        p.data = p.data - lr * p.grad


class Linear:
    def __init__(self, in_features: int, out_features: int, weight_scale: float = 0.01):
        self.W = Tensor(
            np.random.randn(in_features, out_features) * weight_scale,
            requires_grad=True,
        )
        self.b = Tensor(
            np.zeros((out_features,), dtype=self.W.data.dtype), requires_grad=True
        )

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.W) + self.b

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        curr = x
        for layer in self.layers:
            curr = layer(curr)

        return curr


class MSELoss:
    def __init__(self, reduction: str = "mean"):
        """
        MSE Loss

        Args
        ----
        reduction : {"mean","sum"}
            - "mean": L = (1/N) * Σ_i (y_pred_i - y_i)^2   (recommended)
            - "sum" : L = Σ_i (y_pred_i - y_i)^2
        """
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def __call__(self, y_pred: Tensor, y: Tensor) -> Tensor:
        out = (y_pred - y) ** 2
        return out.mean() if self.reduction == "mean" else out.sum()


if __name__ == "__main__":
    #### EXAMPLE TRAINING LOOP!

    # make some linear data
    rng = np.random.default_rng(0)
    N = 256
    x = rng.uniform(-1.0, 1.0, size=(N, 1))

    # mess around with the actual equation at will!
    y = 2.9 * x - 0.7 + 0.05 * rng.normal(size=(N, 1))

    X_t = Tensor(x)  # inputs don't need grads
    Y_t = Tensor(y)  # targets don't need grads

    model = Sequential(
        [
            Linear(in_features=1, out_features=2, weight_scale=0.1),
            Linear(in_features=2, out_features=1, weight_scale=0.1),
        ]
    )

    params = [p for lyr in model.layers for p in lyr.parameters()]

    loss_fn = MSELoss("mean")
    lr = 1e-1
    epochs = 400

    for it in range(epochs):
        # forward pass
        y_pred = model(X_t)
        loss = loss_fn(y_pred, Y_t)

        # backward pass after we zero gradients
        zero_grad(params)
        loss.backwards(np.array(1.0, dtype=loss.data.dtype))

        # update our weights (step)
        sgd(params, lr=lr)

        if it % 100 == 0:
            print(f"iter {it:4d} | loss = {loss.data:.6f}")

    # this is just recovering the effective slope which should match closely to whatever the line is!
    W1 = model.layers[0].W.data  # (1,2)
    b1 = model.layers[0].b.data  # (2,)
    W2 = model.layers[1].W.data  # (2,1)
    b2 = model.layers[1].b.data  # (1,)

    # y = ((x @ W1) + b1) @ W2 + b2 = x @ (W1 @ W2) + (b1 @ W2 + b2)
    W_eff = W1 @ W2  # (1,1)
    b_eff = b1 @ W2 + b2  # (1,)

    a_hat = float(W_eff[0, 0])
    b_hat = float(b_eff[0])

    print("\nEstimated line:")
    print(f"  y ≈ {a_hat:.3f} * x + {b_hat:.3f}   (true: a=2.5, b=-0.7)")
    print(f"Final MSE = {np.mean((x * a_hat + b_hat - y) ** 2):.6f}")
