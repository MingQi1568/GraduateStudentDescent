import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import RandAugment


import imgui
import numpy as np
from sklearn.decomposition import PCA
from imgui.integrations.glfw import GlfwRenderer
import glfw
import OpenGL.GL as gl

import imguiTest

num_atoms = 3
epochs = 1
batch_size = 16
transform = transforms.Compose([
    transforms.ToTensor(),
      # Use a single mean and a single std dev
])
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(1*28*28,10)
        self.linear2 = nn.Linear(10,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)
    
if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=0)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(str(epoch) + ":" + str(running_loss))
    
    data_iter = iter(trainloader)
    images, labels = next(data_iter)
    outputs = model(images).detach().numpy()
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(outputs)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in testloader:
            outputs = model(images)
            loss = criterion(outputs,labels)
            test_loss += loss.item()

            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / len(testloader)
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    window = imguiTest.impl_glfw_init()
    if window is None:
        glfw.terminate()

    # Initialize ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.begin("Hello ImGui!")
        imgui.text("This is some useful text")
        imgui.end()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()