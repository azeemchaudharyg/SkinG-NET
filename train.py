
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn

from models.appnp_model import APPNPGraphClassifier
from models.custom_gnn import CustomSkinCancerGNN
from models.sage_appnp import CustomSAGE_APPNP

from utils.graph_utils import create_fully_connected_edge_index, visualize_graph
from utils.data_loader import load_features_labels, split_data, build_data_loaders

from sklearn.metrics import classification_report, confusion_matrix

import wandb

from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, K, ALPHA, DATA_PATH, WANDB_PROJECT, WANDB_RUN_NAME, WEIGHT_DECAY


# Set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project=WANDB_PROJECT, name='Custom_Results') #WANDB_RUN_NAME
wandb.config.batch_size = BATCH_SIZE
wandb.config.learning_rate = LEARNING_RATE
wandb.config.weight_decay = WEIGHT_DECAY
wandb.config.epochs = EPOCHS

# Load data
features, labels = load_features_labels(DATA_PATH)
node_features, y_all, train_val_idx, test_idx = split_data(features, labels, num_nodes=8) #train_idx, val_idx,

# Build edge index
edge_index = create_fully_connected_edge_index(num_nodes=8)
#visualize_graph(edge_index)

# Data loaders val_loader,
train_loader, test_loader = build_data_loaders(node_features, y_all, edge_index, train_val_idx, test_idx, batch_size=wandb.config.batch_size) #train_idx, val_idx,

# Model
#model = APPNPGraphClassifier(K=K, alpha=ALPHA).to(device)
#model = CustomSkinCancerGNN().to(device)
model = CustomSAGE_APPNP().to(device)


#optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_train_acc = 0.0

# Training
for epoch in range(wandb.config.epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate Training Accuracy
        preds = output.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
        
    avg_epoch_loss = total_loss / len(train_loader)
    train_acc = correct / total

    
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.x, batch.edge_index, batch.batch)
            _, predicted = torch.max(output, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    

    
    wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss, "train_accuracy": train_acc})
    wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss, "val_accuracy": accuracy})
    
    #print(f"Epoch {epoch+1} - loss: {avg_epoch_loss:.4f} - train_acc: {train_acc:.4f}")
    print(f"Epoch {epoch+1} - loss: {avg_epoch_loss:.4f} - val_acc: {accuracy:.4f}")
    
    
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        wandb.log({"val_accuracy": best_val_acc})
    

# Test
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index, batch.batch)
        _, predicted = torch.max(output, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

# Log test accuracy
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
#wandb.log({"test_accuracy": test_acc})

print(f"\nbest_val_acc: {best_val_acc:.4f}\n")

# Classification report and confusion matrix
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, output_dict=True)

# Print
#print("Test Accuracy:", test_acc)
#print("Classification Report:\n", classification_report(all_labels, all_preds))
#print("Confusion Matrix:\n", cm)

# Prepare W&B table
columns = ["class", "precision", "recall", "f1-score", "support"]
table = wandb.Table(columns=columns)

# Log all entries including accuracy, macro avg, and weighted avg
for class_label, metrics in report.items():
    if isinstance(metrics, dict):
        precision = metrics.get("precision", None)
        recall = metrics.get("recall", None)
        f1 = metrics.get("f1-score", None)
        support = metrics.get("support", None)
    else:
        precision = None
        recall = None
        f1 = metrics 
        support = len(all_labels)

    table.add_data(class_label, precision, recall, f1, support)

wandb.log({"classification_report": table})

# Log confusion matrix to wandb
class_names = ["Benign", "Malignant"]
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        preds=all_preds,
        y_true=all_labels,
        class_names=class_names
    )
})


# Create and save the confusion matrix as an image file
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="flare",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the image locally
plt.savefig("confusion_matrix.png")
plt.close()

# Log the saved image to wandb
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})


wandb.finish()
