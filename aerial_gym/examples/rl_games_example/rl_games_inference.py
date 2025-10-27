from collections import OrderedDict
import re
import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = (name or "elu").lower()
    if name in ("elu",):
        return nn.ELU()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("silu", "swish"):  # rl-games uses SiLU naming
        return nn.SiLU()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU()
    if name in ("none", "identity", "linear"):
        return nn.Identity()
    raise ValueError(f"Unsupported MLP activation '{name}' from checkpoint")


class MLP(nn.Module):
    """
    Flexible MLP that constructs its architecture from an RL-Games checkpoint.
    It inspects actor_mlp layer shapes in the checkpoint and builds a matching
    Sequential so that state_dict loading succeeds regardless of hidden sizes
    or activation types.
    """

    def __init__(self, input_dim, output_dim, path, activation: str = "elu"):
        super().__init__()
        self.network = None
        self.activation_name = activation or "elu"
        self._create_and_load_from_checkpoint(path)

    def _create_and_load_from_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model", ckpt)

        # Filter actor weights and map to simple names used by our Sequential
        od_raw = OrderedDict()
        for key, val in sd.items():
            k = str(key)
            # keep only actor branch; drop value/sigma terms
            if "actor_mlp" not in k:
                continue
            if ("value" in k) or ("sigma" in k):
                continue
            # strip RL-Games prefixes to simple names like '0.weight', 'mu.bias'
            k2 = k.replace("a2c_network.actor_mlp.", "")
            k2 = k2.replace("a2c_network.", "")
            od_raw[k2] = val

        # Also include actor output head 'mu'
        for key, val in sd.items():
            k = str(key)
            if k.endswith("mu.weight") or k.endswith("mu.bias"):
                k2 = k.replace("a2c_network.", "")
                od_raw[k2] = val

        # Discover linear layer ids (e.g., '0','2','4',...) and sort them
        linear_ids = []
        for k in od_raw.keys():
            m = re.match(r"^(\d+)\.weight$", k)
            if m:
                linear_ids.append(int(m.group(1)))
        linear_ids = sorted(list(set(linear_ids)))

        if len(linear_ids) == 0 or ("mu.weight" not in od_raw):
            raise RuntimeError("Could not infer actor MLP structure from checkpoint")

        activation = _make_activation(self.activation_name)
        # Build layers according to discovered shapes
        layers = []
        activation_count = 0
        for i, lid in enumerate(linear_ids):
            w_key = f"{lid}.weight"
            b_key = f"{lid}.bias"
            W = od_raw[w_key]
            in_dim = W.shape[1]
            out_dim = W.shape[0]
            lin = nn.Linear(in_dim, out_dim)
            layers.append((f"{lid}", lin))
            if not isinstance(activation, nn.Identity):
                # instantiate a fresh activation module per layer
                act = _make_activation(self.activation_name)
                layers.append((f"act{activation_count+1}", act))
                activation_count += 1
        # Output head 'mu'
        mu_W = od_raw["mu.weight"]
        mu_in = mu_W.shape[1]
        mu_out = mu_W.shape[0]
        mu = nn.Linear(mu_in, mu_out)
        layers.append(("mu", mu))

        self.network = nn.Sequential(OrderedDict(layers))
        # Load weights strictly
        self.network.load_state_dict(od_raw, strict=True)
        layer_names = [name for name, _ in layers]
        print(
            f"Loaded actor MLP from {path} with activation '{self.activation_name}' and layers: {layer_names}"
        )

    def forward(self, x):
        return self.network(x)
