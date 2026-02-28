

# LinkDTI : Drug–Target Interaction via Heterogeneous Graph Learning


## Paper (bioRxiv)
Please read our preprint here:  
https://www.biorxiv.org/content/10.64898/2026.02.21.707210v1

This repo trains a **heterogeneous GCN (DGL)** on a biomedical heterogeneous interaction network (HIN) with node types:
**drug, protein, disease, sideeffect** and multiple relation types (DDI, PPI, DTI, drug–disease, drug–sideeffect, protein–disease).
The model performs **edge prediction** (binary classification) using learned node embeddings and an MLP edge scorer.


---

## 1) Environment / Requirements

### Option A: pip (recommended for quick start)
```bash
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

pip install --upgrade pip
pip install numpy scikit-learn torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html

```
If you find this repo useful for your research, please consider citing our paper.

### BibTeX
```bibtex
@article{LinkDTI2026,
  title   = {LinkDTI: Drug--Target Interaction Prediction via Link Prediction on a Heterogeneous Biomedical Knowledge Graph},
  author  = {<ADD AUTHORS HERE>},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.21.707210},
  url     = {https://www.biorxiv.org/content/10.64898/2026.02.21.707210v1}
}
``
