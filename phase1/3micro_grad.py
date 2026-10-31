# =============================================================================
# phase1/micrograd_from_scratch.py
# =============================================================================
"""
AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND what "autograd" is — and that .backward() is not magic,
     it is ~40 lines of chain rule over a graph. You'll have written it.
  2. SEE the computation graph as an image BEFORE and AFTER training.
     Before: all gradient values (g) are 0.00.
     After:  the same nodes have non-zero gradients — those are the numbers
             that make the model learn.
  3. BE ABLE TO EXPLAIN to someone else, in your own words, why
     "forward, zero, backward, step" is the whole of deep learning training.
  4. HAVE a working micrograd engine you can reuse for any custom loss or
     layer — the same pattern as torch.autograd.Function.
  5. SEE a tiny MLP learn XOR — the smallest problem that requires a
     hidden layer — and understand why a linear model can't solve it.

WHAT YOU ARE DOING (one line)
-----------------------------
  Building PyTorch's .backward() from scratch, training an MLP with it,
  and drawing the graph to see gradients flow.

WHAT YOU WILL GET ON DISK
-------------------------
  outputs/micrograd_before.png   — graph at step 0 (grads are 0)
  outputs/micrograd_after.png    — same graph after training (grads non-zero)

SANITY CHECK (what "success" means here)
----------------------------------------
  - The numeric-vs-autograd gradient check errors below 1e-5 → engine is correct
  - The final XOR loss drops below ~0.05 → training works
  - Predictions land near -1 / +1 in the XOR pattern → model solved it
  If any of these are wrong, the "WHAT FAILURE LOOKS LIKE" block below
  tells you the cause.

WHAT FAILURE LOOKS LIKE
-----------------------
  - max error > 1e-3            → chain-rule bug in Value
  - loss stuck at ~1.0          → LR too low, or grads zeroed
  - loss → nan                  → LR too high (try lr=5.0 to see it)
  - FileNotFoundError on save   → outputs/ missing (this file creates it)
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Reverse-mode AUTODIFF over a computation graph — same idea PyTorch,
#      JAX, and TensorFlow use, in ~40 lines.
#   2. The Value class: wraps a scalar, remembers its parents, and defines
#      a local backward() that propagates dL/d(parent).
#   3. Topological sort → reverse iteration → chain rule. That is the trick.
#   4. A tiny MLP built ONLY from Value objects — no torch — trained with SGD.
#   5. A visual graph so you SEE gradients flow.
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - Every custom op in a VLM/VLA (new attention variant, new action head,
#     new loss) requires a .backward(). This is the pattern.
#   - torch.autograd.Function is literally: define forward, define backward.
#   - π0 (Physical Intelligence) wrote a custom flow-matching action head;
#     its "backward" is this pattern, vectorized.
# =============================================================================

import os
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# Create the outputs dir BEFORE any savefig call.
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# PART 1 — THE AUTOGRAD ENGINE
# =============================================================================
class Value:
    """
    A scalar that remembers how it was created so .backward() can compute
    dL/d(self) for any loss L.

    Attributes
    ----------
    data      : the actual float value
    grad      : dL/d(self); starts at 0.0, filled by .backward()
    _backward : a closure that propagates the local derivative to each parent
    _prev     : the set of Value objects that produced self
    _op       : a short string used only for visual labeling
    """
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        """Topological sort → reverse iteration → chain rule."""
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# =============================================================================
# PART 2 — A TINY NEURAL NET, BUILT ON VALUE
# =============================================================================
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        s = self.b
        for wi, xi in zip(self.w, x):
            s = s + wi * xi
        return s.tanh()
    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, layer_sizes):
        sizes = [n_in] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    def __call__(self, x):
        for L in self.layers: x = L(x)
        return x
    def parameters(self):
        return [p for L in self.layers for p in L.parameters()]


# =============================================================================
# PART 3 — TINY DATASET (2D XOR — no files needed)
# =============================================================================
def make_xor_data():
    X = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    y = [-1.0, 1.0, 1.0, -1.0]
    return X, y


def loss_fn(preds, targets):
    total = Value(0.0)
    for p, t in zip(preds, targets):
        total = total + (p - t) ** 2
    return total * (1.0 / len(preds))


# =============================================================================
# PART 4 — VISUALIZATION
# =============================================================================
def visualize_graph(root, path=f"{OUT_DIR}/micrograd_graph.png", max_nodes=80):
    """
    Draw the computation graph reachable from `root`.
    Yellow = leaf (trainable param), blue = op. Label = op / data / grad.
    """
    G = nx.DiGraph()
    node_data = {}
    order = []

    def visit(v):
        if id(v) in node_data: return
        node_data[id(v)] = v
        order.append(id(v))
        for p in v._prev:
            G.add_edge(id(p), id(v))
            visit(p)

    visit(root)

    if len(order) > max_nodes:
        order = order[-max_nodes:]
        keep = set(order)
        G = G.subgraph(keep).copy()
        node_data = {k: v for k, v in node_data.items() if k in keep}

    labels = {}
    for nid in G.nodes:
        v = node_data[nid]
        op = v._op or "leaf"
        labels[nid] = f"{op}\nd={v.data:+.2f}\ng={v.grad:+.2f}"

    depth = {}
    for nid in G.nodes:
        parents = list(G.predecessors(nid))
        depth[nid] = 0 if not parents else 1 + max(depth.get(p, 0) for p in parents)
    by_depth = {}
    for nid in G.nodes:
        by_depth.setdefault(depth[nid], []).append(nid)
    pos = {}
    for d, nodes in by_depth.items():
        for i, nid in enumerate(nodes):
            pos[nid] = (d, -i + len(nodes) / 2.0)

    fig, ax = plt.subplots(
        figsize=(max(10, len(by_depth) * 1.6),
                 max(6, max(len(v) for v in by_depth.values()) * 0.55))
    )
    colors = ["#ffd27f" if not node_data[n]._prev else "#7fc8ff" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color=colors, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=12, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title("Computation graph  (yellow = leaf/param, blue = op)\n"
                 "d = value, g = gradient", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# PART 5 — MAIN
# =============================================================================
if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    print("=" * 70)
    print("Micrograd — a from-scratch autograd engine")
    print("=" * 70)

    # ---- 1. Numeric gradient check ---------------------------------------
    print("\n[1/5] Verifying autograd vs numeric gradient on a test expression")
    a, b, c = Value(2.0), Value(3.0), Value(-1.5)
    expr = (a * b + c).relu() * a
    expr.backward()
    ag, bg, cg = a.grad, b.grad, c.grad

    eps = 1e-6
    def f(A, B, C):
        r = A * B + C
        r = r if r > 0 else 0.0
        return r * A
    da = (f(2.0 + eps, 3.0, -1.5) - f(2.0 - eps, 3.0, -1.5)) / (2 * eps)
    db = (f(2.0, 3.0 + eps, -1.5) - f(2.0, 3.0 - eps, -1.5)) / (2 * eps)
    dc = (f(2.0, 3.0, -1.5 + eps) - f(2.0, 3.0, -1.5 - eps)) / (2 * eps)
    print(f"  autograd:  da={ag:+.6f}  db={bg:+.6f}  dc={cg:+.6f}")
    print(f"  numeric :  da={da:+.6f}  db={db:+.6f}  dc={dc:+.6f}")
    print(f"  max error: {max(abs(ag-da), abs(bg-db), abs(cg-dc)):.2e}  (should be <1e-5)")

    # ---- 2. Build MLP and data -------------------------------------------
    print("\n[2/5] Building 2 → 4 → 1 MLP on XOR")
    mlp = MLP(2, [4, 1])
    params = mlp.parameters()
    print(f"  total parameters: {len(params)}")
    X, y = make_xor_data()

    # ---- 3. Visualize ONE sample's graph ---------------------------------
    print("\n[3/5] Visualizing the graph for ONE forward pass (sample 0 only)")
    pred0 = mlp(X[0])
    loss_single = (pred0 - y[0]) ** 2
    visualize_graph(loss_single, path=f"{OUT_DIR}/micrograd_before.png")
    print(f"  loss on sample 0 = {loss_single.data:.4f}  (before training)")

    # ---- 4. Train with SGD -----------------------------------------------
    print("\n[4/5] Training (SGD, lr=0.1, 200 steps)")
    lr = 0.1
    for step in range(200):
        preds = [mlp(x) for x in X]
        loss = loss_fn(preds, y)
        for p in params: p.grad = 0.0
        loss.backward()
        for p in params: p.data -= lr * p.grad
        if step % 25 == 0 or step == 199:
            print(f"  step {step:3d}  loss = {loss.data:.4f}")

    # ---- 5. Visualize AFTER + report -------------------------------------
    print("\n[5/5] Visualizing the graph after training (sample 0 again)")
    pred0 = mlp(X[0])
    loss_single = (pred0 - y[0]) ** 2
    for p in params: p.grad = 0.0
    loss_single.backward()
    visualize_graph(loss_single, path=f"{OUT_DIR}/micrograd_after.png")
    print(f"  loss on sample 0 = {loss_single.data:.4f}  (after training)")

    preds = [mlp(x) for x in X]
    print("\n  predictions (tanh output, target in [-1, +1]):")
    for x, p, t in zip(X, preds, y):
        print(f"    x=({x[0].data:.1f},{x[1].data:.1f})  pred={p.data:+.3f}  target={t:+.3f}")

    print(f"\nAll plots saved to ./{OUT_DIR}/micrograd_*.png")