import pickle
import npnet as nn


class Saver:
    @staticmethod
    def save(model, path):
        assert isinstance(model, nn.Module)
        vars = {name: p["vars"] for name, p in model.params.items()}
        with open(path, "wb") as f:
            pickle.dump(vars, f)

    @staticmethod
    def restore(model, path):
        assert isinstance(model, nn.Module)
        with open(path, "rb") as f:
            params = pickle.load(f)
        for name, param in params.items():
            for p_name in model.params[name]["vars"].keys():
                model.params[name]["vars"][p_name][:] = param[p_name]
                model.params[name]["vars"][p_name][:] = param[p_name]
