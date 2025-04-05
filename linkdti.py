import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.preprocessing import normalize
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    

    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Enforce deterministic algorithms
    torch.set_num_threads(1)
    print(f"Seed set to: {seed}")

set_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Step 1: Load Data
def load_data():
    print("Loading data...")
    network_path = '../data/'
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    true_drug = 708
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_chemical = drug_chemical[:true_drug, :true_drug]
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    num_drug = len(drug_drug)
    num_protein = len(protein_protein)

    # Removed the self-loop
    drug_chemical = drug_chemical - np.identity(num_drug)
    protein_sequence = protein_sequence / 100.
    protein_sequence = protein_sequence - np.identity(num_protein)

    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')
    drug_protein = np.loadtxt(network_path + 'mat_drug_protein_homo_protein_drug.txt')
    

    print("Data loaded successfully.")
    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, drug_protein

drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = load_data()

drug_similarity = np.loadtxt('../data/Similarity_Matrix_Drugs.txt')
protein_similarity = np.loadtxt('../data/Similarity_Matrix_Proteins.txt')

# Normalize the similarity matrices
drug_similarity = normalize(drug_similarity, norm='l1', axis=1)
protein_similarity = normalize(protein_similarity, norm='l1', axis=1)

# Step 2: Construct Graph
def construct_graph(drug_drug, drug_disease, drug_sideeffect, protein_protein, protein_disease, drug_protein):
    print("Constructing graph...")
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = [(i, i) for i in range(num_drug)]
    list_protein = [(i, i) for i in range(num_protein)]
    list_disease = [(i, i) for i in range(num_disease)]
    list_sideeffect = [(i, i) for i in range(num_sideeffect)]

    list_DDI = [(row, col) for row in range(num_drug) for col in range(num_drug) if drug_drug[row, col] > 0]
    list_PPI = [(row, col) for row in range(num_protein) for col in range(num_protein) if protein_protein[row, col] > 0]

    list_drug_protein = [(row, col) for row in range(num_drug) for col in range(num_protein) if drug_protein[row, col] > 0]
    list_protein_drug = [(col, row) for row in range(num_drug) for col in range(num_protein) if drug_protein[row, col] > 0]

    list_drug_sideeffect = [(row, col) for row in range(num_drug) for col in range(num_sideeffect) if drug_sideeffect[row, col] > 0]
    list_sideeffect_drug = [(col, row) for row in range(num_drug) for col in range(num_sideeffect) if drug_sideeffect[row, col] > 0]

    list_drug_disease = [(row, col) for row in range(num_drug) for col in range(num_disease) if drug_disease[row, col] > 0]
    list_disease_drug = [(col, row) for row in range(num_drug) for col in range(num_disease) if drug_disease[row, col] > 0]

    list_protein_disease = [(row, col) for row in range(num_protein) for col in range(num_disease) if protein_disease[row, col] > 0]
    list_disease_protein = [(col, row) for row in range(num_protein) for col in range(num_disease) if protein_disease[row, col] > 0]

    g_HIN = dgl.heterograph({
        ('disease', 'disease_disease virtual', 'disease'): list_disease,
        ('drug', 'drug_drug virtual', 'drug'): list_drug,
        ('protein', 'protein_protein virtual', 'protein'): list_protein,
        ('sideeffect', 'sideeffect_sideeffect virtual', 'sideeffect'): list_sideeffect,
        ('drug', 'drug_drug interaction', 'drug'): list_DDI,
        ('protein', 'protein_protein interaction', 'protein'): list_PPI,
        ('drug', 'drug_protein interaction', 'protein'): list_drug_protein,
        ('protein', 'protein_drug interaction', 'drug'): list_protein_drug,
        ('drug', 'drug_sideeffect association', 'sideeffect'): list_drug_sideeffect,
        ('sideeffect', 'sideeffect_drug association', 'drug'): list_sideeffect_drug,
        ('drug', 'drug_disease association', 'disease'): list_drug_disease,
        ('disease', 'disease_drug association', 'drug'): list_disease_drug,
        ('protein', 'protein_disease association', 'disease'): list_protein_disease,
        ('disease', 'disease_protein association', 'protein'): list_disease_protein
    })

    g = g_HIN.edge_type_subgraph([
        'drug_drug interaction', 'protein_protein interaction',
        'drug_protein interaction', 'protein_drug interaction',
        'drug_sideeffect association', 'sideeffect_drug association',
        'drug_disease association', 'disease_drug association',
        'protein_disease association', 'disease_protein association'
    ])

    print("Graph construction done.")
    return g

g = construct_graph(drug_d, drug_di, drug_side, protein_p, protein_di, dti_original)
g = g.to(device)

# Step 3: Initialize Features
def initialize_features(g, dim_embedding=512, seed=42):
    print("Initializing features...")
    set_seed(seed)
    features = {}
    for ntype in g.ntypes:
        num_nodes = g.number_of_nodes(ntype)
        feat = nn.Parameter(torch.FloatTensor(num_nodes, dim_embedding).to(device))
        torch.nn.init.normal_(feat, mean=0, std=0.1)
        features[ntype] = feat
    return features

features = initialize_features(g)


# Step 4: Creating samples and labels for each interaction type
def create_samples_and_labels(interaction_matrix, src_type, dst_type, num_neg_samples_per_pos=1, undirected=True, seed=42):
    print(f"Creating samples and labels for {src_type} -> {dst_type} interactions...")
    set_seed(seed)
    whole_positive_index = set()
    whole_negative_index = []

    for i in range(np.shape(interaction_matrix)[0]):
        for j in range(np.shape(interaction_matrix)[1]):
            if undirected:
                pair = (min(i, j), max(i, j))
            else:
                pair = (i, j)
                
            if int(interaction_matrix[i][j]) == 1:
                whole_positive_index.add(pair)
            elif int(interaction_matrix[i][j]) == 0:
                whole_negative_index.append(pair)

    num_neg_samples = min(num_neg_samples_per_pos * len(whole_positive_index), len(whole_negative_index))

    negative_sample_index = np.random.choice(len(whole_negative_index), size=num_neg_samples, replace=False)
    negative_sample_pairs = [whole_negative_index[idx] for idx in negative_sample_index]

    data_set = np.zeros((num_neg_samples + len(whole_positive_index), 3), dtype=np.float32)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_pairs:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 0
        count += 1

    samples = torch.tensor(data_set[:, :2], dtype=torch.long).to(device)  # Move to device
    labels = torch.tensor(data_set[:, 2], dtype=torch.float).to(device) 
    return samples, labels

edge_pairs_dict = {
    'drug_protein': torch.tensor(create_samples_and_labels(dti_original, 'drug', 'protein', undirected=True)[0], dtype=torch.long),
    'protein_protein': torch.tensor(create_samples_and_labels(protein_p, 'protein', 'protein', undirected=True)[0], dtype=torch.long),
    'drug_drug': torch.tensor(create_samples_and_labels(drug_d, 'drug', 'drug', undirected=True)[0], dtype=torch.long),
    'drug_sideeffect': torch.tensor(create_samples_and_labels(drug_side, 'drug', 'sideeffect', undirected=False)[0], dtype=torch.long),
    'drug_disease': torch.tensor(create_samples_and_labels(drug_di, 'drug', 'disease', undirected=False)[0], dtype=torch.long),
    'protein_disease': torch.tensor(create_samples_and_labels(protein_di, 'protein', 'disease', undirected=False)[0], dtype=torch.long)
}

labels_dict = {
    'drug_protein': torch.tensor(create_samples_and_labels(dti_original, 'drug', 'protein', undirected=True)[1], dtype=torch.float),
    'protein_protein': torch.tensor(create_samples_and_labels(protein_p, 'protein', 'protein', undirected=True)[1], dtype=torch.float),
    'drug_drug': torch.tensor(create_samples_and_labels(drug_d, 'drug', 'drug', undirected=True)[1], dtype=torch.float),
    'drug_sideeffect': torch.tensor(create_samples_and_labels(drug_side, 'drug', 'sideeffect', undirected=False)[1], dtype=torch.float),
    'drug_disease': torch.tensor(create_samples_and_labels(drug_di, 'drug', 'disease', undirected=False)[1], dtype=torch.float),
    'protein_disease': torch.tensor(create_samples_and_labels(protein_di, 'protein', 'disease', undirected=False)[1], dtype=torch.float)
}

# Step 5: Define HeteroGCN Layer
class HeteroGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu):
        super(HeteroGCNLayer, self).__init__()
        self.conv = dgl.nn.HeteroGraphConv({
            'drug_drug interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_protein interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_protein interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_drug interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_disease association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'disease_drug association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_sideeffect association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'sideeffect_drug association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_disease association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'disease_protein association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True)
        }, aggregate='sum')
        self.activation = activation

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        for k, v in h.items():
            if self.activation:
                h[k] = self.activation(v)
        return h

class MultiHopHeteroGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_hops=1, activation=F.relu):
        super(MultiHopHeteroGCNLayer, self).__init__()
        self.num_hops = num_hops
        self.conv = dgl.nn.HeteroGraphConv({
            'drug_drug interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_protein interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_protein interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_drug interaction': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_disease association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'disease_drug association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'drug_sideeffect association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'sideeffect_drug association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'protein_disease association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True),
            'disease_protein association': dgl.nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True, weight=True)
        }, aggregate='sum')
        self.activation = activation

    def forward(self, g, inputs):
        h = inputs
        for _ in range(self.num_hops):
            h = self.conv(g, h)
            for k, v in h.items():
                if self.activation:
                    h[k] = self.activation(v)
        return h


# Step 6: Define the HeteroGCN Model
class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HeteroGCN, self).__init__()
        self.layer1 = HeteroGCNLayer(in_feats, hidden_feats)
        self.layer2 = HeteroGCNLayer(hidden_feats, out_feats)
        self.edge_predictor = nn.Sequential(
            nn.Linear(out_feats * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, g, features, edge_pairs, etype):
        h = self.layer1(g, features)
        h = self.layer2(g, h)
        
        # Gather node embeddings for the specific edge type
        src_embeddings = h[etype[0]][edge_pairs[:, 0]]
        dst_embeddings = h[etype[2]][edge_pairs[:, 1]]
        
        # Edge prediction
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return self.edge_predictor(edge_features).squeeze()

# Step 7: Stratified Split Function
def stratified_split(edge_pairs, labels, val_size=0.1, test_size=0.1, seed=42):
    set_seed(seed)
    # Move tensors to CPU and convert to NumPy arrays for sklearn compatibility
    edge_pairs = edge_pairs.cpu().numpy()
    labels = labels.cpu().numpy()
    # Split the data into train+val and test
    train_val_pairs, test_pairs, train_val_labels, test_labels = train_test_split(
        edge_pairs, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    
    # Split train+val into train and val
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        train_val_pairs, train_val_labels, test_size=val_size/(1-test_size), stratify=train_val_labels, random_state=seed
    )
    
    return (
        torch.tensor(train_pairs, dtype=torch.long).to(device),
        torch.tensor(val_pairs, dtype=torch.long).to(device),
        torch.tensor(test_pairs, dtype=torch.long).to(device),
        torch.tensor(train_labels, dtype=torch.float).to(device),
        torch.tensor(val_labels, dtype=torch.float).to(device),
        torch.tensor(test_labels, dtype=torch.float).to(device),
    )

# Step 8: Initialize Model, Optimizer, and Loss Function
in_feats = 512
hidden_feats = 256
out_feats = 128
model = HeteroGCN(in_feats, hidden_feats, out_feats)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Step 9: Training Loop with Early Stopping and Test AUROC Calculation
# Updated training loop to save results to a file
def train_hetero_gcn(g, features, edge_pairs_dict, labels_dict, canonical_edge_type_dict, epochs=100, patience=10):
    g = g.to(device)
    for ntype in g.ntypes:
        features[ntype] = features[ntype].to(device)
    
    # Ensure model is on the correct device
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0

    # Split data into train, val, test
    train_pairs_dict, val_pairs_dict, test_pairs_dict = {}, {}, {}
    train_labels_dict, val_labels_dict, test_labels_dict = {}, {}, {}

    for edge_type, edge_pairs in edge_pairs_dict.items():
        labels = labels_dict[edge_type]
        train_pairs, val_pairs, test_pairs, train_labels, val_labels, test_labels = stratified_split(
            edge_pairs, labels, val_size=0.1, test_size=0.1)
        train_pairs_dict[edge_type] = train_pairs
        val_pairs_dict[edge_type] = val_pairs
        test_pairs_dict[edge_type] = test_pairs
        train_labels_dict[edge_type] = train_labels
        val_labels_dict[edge_type] = val_labels
        test_labels_dict[edge_type] = test_labels

    with open('new_Het_results.txt', 'w') as f:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for edge_type, edge_pairs in train_pairs_dict.items():
                labels = train_labels_dict[edge_type]
                canonical_etype = canonical_edge_type_dict[edge_type]
                optimizer.zero_grad()
                predictions = model(g, features, edge_pairs, canonical_etype)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            val_loss, val_auroc, val_auprc, _, val_conf_matrix = evaluate_model_with_confusion_matrix(
                model, g, features, val_pairs_dict, val_labels_dict, canonical_edge_type_dict)

            f.write(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}, Validation AUPRC: {val_auprc:.4f}\n")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    f.write("Early stopping triggered.\n")
                    break

        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_loss, test_auroc, test_auprc, mispredictions, conf_matrix = evaluate_model_with_confusion_matrix(
            model, g, features, test_pairs_dict, test_labels_dict, canonical_edge_type_dict)

        f.write(f"Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")

        # Log mispredictions
        f.write("Mispredictions:\n")
        for src, dst, true_label, predicted_label in mispredictions:
            f.write(f"Edge ({src}, {dst}), True Label: {true_label}, Predicted Label: {predicted_label}\n")
        print("Results saved to new_Het_results.txt.")



import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np

# Helper function for AUPRC calculation
def calculate_auprc(labels, predictions):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return average_precision_score(labels, predictions)

# Updated evaluation function with AUPRC calculation and misprediction logging
from sklearn.metrics import confusion_matrix

# Updated evaluation function with confusion matrix calculation
def evaluate_model_with_confusion_matrix(model, g, features, edge_pairs_dict, labels_dict, canonical_edge_type_dict):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    all_predictions = []
    all_labels = []
    all_pairs = []  # To track the pairs being predicted
    mispredictions = []

    with torch.no_grad():
        for edge_type, edge_pairs in edge_pairs_dict.items():
            canonical_etype = canonical_edge_type_dict[edge_type]
            labels = labels_dict[edge_type]
            predictions = model(g, features, edge_pairs, canonical_etype)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_pairs.extend(edge_pairs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_pairs = np.array(all_pairs)

    # AUPRC Calculation
    auprc = calculate_auprc(all_labels, all_predictions)

    # Identify mispredictions
    threshold = 0.5  # Binary classification threshold
    predicted_classes = (all_predictions >= threshold).astype(int)
    for pair, true_label, predicted_label in zip(all_pairs, all_labels, predicted_classes):
        if true_label != predicted_label:
            mispredictions.append((pair[0], pair[1], true_label, predicted_label))

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, predicted_classes)

    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_predictions)
    else:
        auroc = float('nan')
        print("Warning: Only one class present in y_true. ROC AUC score is not defined.")

    return total_loss, auroc, auprc, mispredictions, conf_matrix



# Step 11: Define Canonical Edge Types
canonical_edge_type_dict = {
    'drug_protein': ('drug', 'drug_protein interaction', 'protein'),
    'protein_protein': ('protein', 'protein_protein interaction', 'protein'),
    'drug_drug': ('drug', 'drug_drug interaction', 'drug'),
    'drug_sideeffect': ('drug', 'drug_sideeffect association', 'sideeffect'),
    'drug_disease': ('drug', 'drug_disease association', 'disease'),
    'protein_disease': ('protein', 'protein_disease association', 'disease')
}

# Step 12: Train the Model
train_hetero_gcn(g, features, edge_pairs_dict, labels_dict, canonical_edge_type_dict, epochs=10, patience=10)
