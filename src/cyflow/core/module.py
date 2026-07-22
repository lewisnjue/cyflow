from typing import Any, List


class Module:
    def __init__(self):
        """Base module class for all neural network components."""
        self.training = True

    def _iter_children(self):
        for _, value in vars(self).items():
            yield value

    def apply(self, fn):
        """Apply a function to this module, its tensors, and nested submodules."""
        fn(self)

        for name, value in vars(self).items():
            if hasattr(value, "data"):
                setattr(self, name, fn(value))
            elif isinstance(value, Module):
                value.apply(fn)
            elif isinstance(value, (list, tuple)):
                new_items = []
                for item in value:
                    if isinstance(item, Module):
                        item.apply(fn)
                        new_items.append(item)
                    elif isinstance(item, (list, tuple)):
                        nested_items = []
                        for nested_item in item:
                            if isinstance(nested_item, Module):
                                nested_item.apply(fn)
                                nested_items.append(nested_item)
                            elif hasattr(nested_item, "data"):
                                nested_items.append(fn(nested_item))
                            else:
                                nested_items.append(nested_item)
                        new_items.append(tuple(nested_items) if isinstance(item, tuple) else nested_items)
                    elif hasattr(item, "data"):
                        new_items.append(fn(item))
                    else:
                        new_items.append(item)
                if isinstance(value, list):
                    setattr(self, name, new_items)
                else:
                    setattr(self, name, tuple(new_items))
            elif isinstance(value, dict):
                updated = {}
                for key, item in value.items():
                    if isinstance(item, Module):
                        item.apply(fn)
                        updated[key] = item
                    elif hasattr(item, "data"):
                        updated[key] = fn(item)
                    else:
                        updated[key] = item
                setattr(self, name, updated)

        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params: List[Any] = []
        for value in self._iter_children():
            if hasattr(value, "data"):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, (list, tuple)):
                        for nested_item in item:
                            if isinstance(nested_item, Module):
                                params.extend(nested_item.parameters())
                            elif hasattr(nested_item, "data"):
                                params.append(nested_item)
                    elif hasattr(item, "data"):
                        params.append(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif hasattr(item, "data"):
                        params.append(item)

        return params

    def train(self) -> None:
        self.training = True
        for value in self._iter_children():
            if isinstance(value, Module):
                value.train()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train()

    def inference_mode(self):
        self.eval()
        return

    def eval(self) -> None:
        self.training = False
        for value in self._iter_children():
            if isinstance(value, Module):
                value.eval()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.eval()
