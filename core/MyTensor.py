import math
class MyTensor:
    def __init__(self, data, require_grad = False, _childs=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self.require_grad = require_grad

        self._backward = lambda: None
        self._childs = set(_childs)
        self._op = _op
    
    def __repr__(self):
        return f"MyTensor(data={self.data}, require_grad ={self.require_grad}, grad={self.grad})"
    
    def __add__(self, other):
        # Addition operation
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        out = MyTensor(self.data + other.data, self.require_grad or other.require_grad, _childs=(self, other), _op='+')
        def _backward():
            if self.require_grad:
                self.grad += out.grad
            if other.require_grad:
                other.grad += out.grad 
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        # Right addition to support scalar + MyTensor
        return self + other
      
    def __mul__(self, other):
        # Multiplication operation
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        out = MyTensor(self.data * other.data, self.require_grad or other.require_grad, _childs=(self, other), _op='*')
        def _backward():
            if self.require_grad:
                self.grad += (other.data * out.grad)
            if other.require_grad:
                other.grad += (self.data * out.grad)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        # Right multiplication to support scalar * MyTensor
        return self * other
    
    def __pow__(self, other):
        # Power operation
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        out = MyTensor(self.data ** other.data, require_grad = self.require_grad or other.require_grad, _childs=(self, other), _op='pow')
        def _backward():
            if self.require_grad:
                self.grad += (other.data * (self.data ** (other.data - 1)) * out.grad)
            if other.require_grad:
                other.grad += ((self.data ** other.data) * math.log(self.data) * out.grad)
        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        # Right power to support scalar ** MyTensor
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return other ** self

    def __neg__(self):
        # Negation operation
        return self * -1
    
    def __sub__(self, other):
        # Subtraction operation
        return self + (-other)

    def __rsub__(self, other):
        # Right subtraction to support scalar - MyTensor
        return -(self - other)
    
    def __truediv__(self, other):
        # Division operation
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        out = MyTensor(self.data / other.data, self.require_grad or other.require_grad, _childs=(self, other), _op='/')
        def _backward():
            if self.require_grad:
                self.grad += (out.grad / other.data)
            if other.require_grad:
                other.grad += ((-self.data / (other.data ** 2)) * out.grad)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # Right division to support scalar / MyTensor
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return other / self
    
    def exp(self):
        # Exponential operation
        out = MyTensor(data = math.exp(self.data), require_grad=self.require_grad, _childs=(self,), _op='exp')
        def _backward():
            if self.require_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # Implementing tanh activation function
        out = math.exp(2*self.data) - 1 / (math.exp(2*self.data) + 1)
        out = MyTensor(out, require_grad=self.require_grad, _childs=(self,), _op='tanh')
        def _backward():
            if self.require_grad:
                self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Build topological order of nodes and call backward on them
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._childs:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()