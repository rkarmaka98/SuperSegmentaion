import torch
from torchsummary import summary
from torchviz import make_dot
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2

def main():
    # Define a fixed input size for visualization
    input_size = (1, 256, 512)  # [channels, height, width]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SuperPointNet_gauss2 with input size: {input_size}")
    model = SuperPointNet_gauss2(num_classes=1).to(device)
    model.eval()

    # Print model summary
    print(f"\n--- SuperPointNet_gauss2 Summary ---\n")
    summary(model, input_size=input_size, device=str(device))

    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)

    # Forward pass
    output = model(dummy_input)

    # Visualize only the segmentation head in detail
    dot = make_dot(output["segmentation"], 
                   params=dict(model.named_parameters()),
                   show_attrs=False, show_saved=False)

    dot.format = 'png'
    dot.render("superpoint_segmentation_diagram", view=False)
    print("Saved model diagram as superpoint_segmentation_diagram.png")

if __name__ == "__main__":
    main()
