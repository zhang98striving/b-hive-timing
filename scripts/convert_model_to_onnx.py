import argparse
import numpy as np
import torch
import onnx

from utils.models.deepjet import DeepJet


def load_model(model_path: str, config_dict: str):
    print(f"loading model {model_path}")

    config_dict = np.load(config_dict, allow_pickle=True).item()
    model = DeepJet(config_dict["model"]["feature_edges"]).to("cpu")
    best_model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(best_model["model_state_dict"])
    model.eval()
    return model, None


def save_to_onnx(model, input_shape, output_path: str):
    print(f"saving to {output_path}")
    x = torch.randn(1000, 613, 1, requires_grad=True)
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        output_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def main(model_path, config_dict, output_path):
    model, input_shape = load_model(model_path, config_dict)
    save_to_onnx(model, input_shape, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Path to model.")
    parser.add_argument("--config_dict", "-c", type=str, help="Path to config dict.")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    args = parser.parse_args()
    main(args.model, args.config_dict, args.output)
