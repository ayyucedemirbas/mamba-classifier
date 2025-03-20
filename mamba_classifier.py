import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class SimpleMambaLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(L):
            u_t = x[:, t]
            g_t = torch.sigmoid(self.gate_proj(u_t))
            h = (1 - g_t) * h + g_t * u_t
            outputs.append(h.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)
        out_seq = self.out_proj(out_seq)
        return out_seq


class VisionMambaClassifier(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64,
                 num_layers=4, num_classes=100):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.mamba_layers = nn.ModuleList([
            SimpleMambaLayer(embed_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class):

        self.model.zero_grad()
        output = self.model(input_image)
        score = output[0, target_class]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, H', W')
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize between 0 and 1
        return cam

def show_cam_on_image(img, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    overlayed_img = heatmap * alpha + img
    overlayed_img = overlayed_img / np.max(overlayed_img)
    return np.uint8(255 * overlayed_img)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Epoch [{epoch}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%')
    return test_loss, test_acc


def main():
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VisionMambaClassifier(img_size=32, patch_size=4, in_chans=3, embed_dim=64,
                                  num_layers=4, num_classes=100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
    
    torch.save(model.state_dict(), 'vision_mamba_cifar100.pth')
    print("Model saved as vision_mamba_cifar100.pth")

    gradcam = GradCAM(model, model.patch_embed.proj)

    sample_img, label = test_dataset[0]
    sample_img_unsqueezed = sample_img.unsqueeze(0).to(device)

    model.eval()
    output = model(sample_img_unsqueezed)
    pred_class = output.argmax(dim=1).item()
    print(f"True label: {label}, Predicted label: {pred_class}")

    cam = gradcam.generate_cam(sample_img_unsqueezed, target_class=pred_class)

    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    img_np = sample_img.cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    overlayed_img = show_cam_on_image(img_np, cam, alpha=0.5)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed_img)
    plt.title(f"GradCAM for class {pred_class}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
