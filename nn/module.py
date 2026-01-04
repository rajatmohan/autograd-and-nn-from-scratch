class Module:
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, 'require_grad') and item.require_grad:
                        params.append(item)
            elif hasattr(attr, 'require_grad') and attr.require_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args):
        raise NotImplementedError