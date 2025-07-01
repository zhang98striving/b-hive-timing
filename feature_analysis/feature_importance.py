import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

def get_feature_lists(model):
    # Get feature names automatically
    cpf = getattr(model, "cpf_candidates", None)
    npf = getattr(model, "npf_candidates", None)
    vtx = getattr(model, "vtx_features", None)
    glob = getattr(model, "global_features", None)
    return cpf, npf, vtx, glob

def compute_importance(model, dataloader, device, target_class=0, max_batches=10):
    model.eval()
    ig = IntegratedGradients(model)
    cpf, npf, vtx, glob = get_feature_lists(model)

    sum_attr_global = torch.zeros(len(glob), device=device)
    sum_attr_cpf = torch.zeros(len(cpf), device=device)
    sum_attr_npf = torch.zeros(len(npf), device=device)
    sum_attr_vtx = torch.zeros(len(vtx), device=device)
    n_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            global_features = batch[0].clone().detach().float().to(device)
            cpf_features = batch[1].clone().detach().float().to(device)
            npf_features = batch[2].clone().detach().float().to(device)
            vtx_features = batch[3].clone().detach().float().to(device)
        else:
            raise RuntimeError("Unknown batch format!")

        # Add batch dimension if needed
        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
            cpf_features = cpf_features.unsqueeze(0)
            npf_features = npf_features.unsqueeze(0)
            vtx_features = vtx_features.unsqueeze(0)

        baseline_global = torch.zeros_like(global_features)
        baseline_cpf = torch.zeros_like(cpf_features)
        baseline_npf = torch.zeros_like(npf_features)
        baseline_vtx = torch.zeros_like(vtx_features)

        try:
            attributions = ig.attribute(
                inputs=(global_features, cpf_features, npf_features, vtx_features),
                baselines=(baseline_global, baseline_cpf, baseline_npf, baseline_vtx),
                target=target_class,
            )
        except Exception as e:
            print("IG failed, check model and input:", e)
            continue

        attr_global, attr_cpf, attr_npf, attr_vtx = attributions

        # Average over batch and particle dimensions
        sum_attr_global += attr_global.abs().mean(dim=0)
        sum_attr_cpf += attr_cpf.abs().mean(dim=(0, 1))
        sum_attr_npf += attr_npf.abs().mean(dim=(0, 1))
        sum_attr_vtx += attr_vtx.abs().mean(dim=(0, 1))
        n_samples += 1

    # Calculate mean importance
    mean_attr_global = (sum_attr_global / n_samples).cpu().numpy()
    mean_attr_cpf = (sum_attr_cpf / n_samples).cpu().numpy()
    mean_attr_npf = (sum_attr_npf / n_samples).cpu().numpy()
    mean_attr_vtx = (sum_attr_vtx / n_samples).cpu().numpy()

    return mean_attr_global, mean_attr_cpf, mean_attr_npf, mean_attr_vtx, glob, cpf, npf, vtx

def plot_importance(importance, names, title, save_path=None):
    # Calculate figure size based on number of features
    fig_height = max(6, len(names) * 0.3)
    fig_width = 8
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create horizontal bar plot
    y_positions = range(len(names))
    plt.barh(y_positions, importance)
    
    # Set labels
    plt.yticks(y_positions, names)
    plt.title(title)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def get_feature_importance_by_name(importance_dict, feature_name):
    """
    Get importance value for a specific feature name
    Example: get_feature_importance_by_name(importance_dict, "Jet_rel_time")
    """
    for category, (values, names) in importance_dict.items():
        if feature_name in names:
            idx = names.index(feature_name)
            return values[idx], category
    return None, None

def print_top_features(importance_dict, top_k=10):
    """
    Print top_k most important features for each category
    """
    for category, (values, names) in importance_dict.items():
        print(f"\n=== {category.upper()} Features (Top {top_k}) ===")
        feature_pairs = list(zip(values, names))
        feature_pairs.sort(key=lambda x: x[0], reverse=True)
        
        for i, (importance, name) in enumerate(feature_pairs[:top_k]):
            print(f"{i+1:2d}. {name:30s}: {importance:.6f}")

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    from utils.models.deepjet import DeepJet
    model = DeepJet()
    
    # Load model weights
    best_model_path = (
        "/net/data_cms3a-1/zhang/b-hive-2025/Timing_PU02/TrainingTask/offline_run3"
        "/test_ttbar_PU200_weightedtimingPU/test_ttbar_PU200_weightedtimingPU_training00"
        "/DeepJet/epochs_150/nominal/best_model.pt"
    )
    
    try:
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Best DeepJet model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using random weights instead.")
    
    model.to(device)

    # Load data
    from utils.torch.deepJetDataset import DeepJetDataset
    
    bins_pt = [10, 25, 30, 35, 40, 45, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 2001]
    bins_eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1, 1.5, 2.0, 2.51]
    
    val_dataset = DeepJetDataset(
        files=["/net/data_cms3a-1/zhang/b-hive-2025/Timing_PU02/DatasetConstructorTask/offline_run3/test_ttbar_PU200_weightedtimingPU_test/file_0.npz"],
        model=model,
        data_type="validation",
        bins_pt=bins_pt,
        bins_eta=bins_eta
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    # Compute feature importance
    mean_attr_global, mean_attr_cpf, mean_attr_npf, mean_attr_vtx, glob, cpf, npf, vtx = compute_importance(
        model, val_loader, device, target_class=0, max_batches=10
    )

    # Organize results
    importance_dict = {
        "global": (mean_attr_global, glob),
        "cpf": (mean_attr_cpf, cpf),
        "npf": (mean_attr_npf, npf),
        "vtx": (mean_attr_vtx, vtx)
    }
    
    # Output results
    import os
    
    os.makedirs("./DeepJet/", exist_ok=True)
    
    output_text = []
    output_text.append("\n" + "="*80)
    output_text.append("FEATURE IMPORTANCE ANALYSIS RESULTS")
    output_text.append("="*80)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS RESULTS")
    print("="*80)
    
    for category, (values, names) in importance_dict.items():
        output_text.append(f"\n{category.upper()} FEATURES:")
        output_text.append("-" * 50)
        
        print(f"\n{category.upper()} FEATURES:")
        print("-" * 50)
        
        feature_pairs = list(zip(values, names))
        feature_pairs.sort(key=lambda x: x[0], reverse=True)
        
        for i, (importance, name) in enumerate(feature_pairs):
            line = f"{i+1:2d}. {name:35s}: {importance:.6f}"
            output_text.append(line)
            print(line)
    
    # Save to file
    with open("./DeepJet/feature_importance_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_text))
    
    print(f"\nResults saved to: ./DeepJet/feature_importance_results.txt")
    
    # Example: query specific feature importance
    # importance_value, category = get_feature_importance_by_name(importance_dict, "Jet_rel_time")
    # if importance_value is not None:
    #     print(f"Jet_rel_time importance: {importance_value:.6f} (in {category} features)")
    # else:
    #     print("Jet_rel_time not found")

    # Print top features
    # print_top_features(importance_dict, top_k=5)

    # Generate plots
    plot_importance(mean_attr_global, glob, "Global Features Importance", "./DeepJet/global_importance.png")
    plot_importance(mean_attr_cpf, cpf, "CPF Features Importance", "./DeepJet/cpf_importance.png")
    plot_importance(mean_attr_npf, npf, "NPF Features Importance", "./DeepJet/npf_importance.png")
    plot_importance(mean_attr_vtx, vtx, "VTX Features Importance", "./DeepJet/vtx_importance.png")