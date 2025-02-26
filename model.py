import torch
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.hooks = []
        
        def hook_fn(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradient)

        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(hook_fn))

        output = self.model(x)
        return output

    def generate_cam(self, class_idx):
        self.model.zero_grad()
        class_loss = self.activations[0, class_idx].sum()
        class_loss.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations were not properly computed.")
        
        gradients = self.gradients
        activations = self.activations
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap - torch.min(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.cpu().detach().numpy()
        return heatmap

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Function to display image and heatmap
def show_heatmap_on_image(heatmap, image, alpha=0.4):
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    
    # Convert image to numpy array format
    image_np = np.array(image)
    
    # Apply heatmap on image
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, image_np, 1 - alpha, 0)
    
    # Display image with heatmap
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Image with Heatmap')
    plt.axis('off')
    
    plt.show()
