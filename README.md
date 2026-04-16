# BoxGCN on Amazon Pet Supplies

This repository contains a PyTorch implementation of `GateBoxGCN` for implicit-feedback recommendation.

The main training entry point is:

```bash
code/train.py
```

## Project Structure

```text
BoxGCN/
├── code/
│   ├── main.py
│   ├── data_utils.py
│   └── evaluate_v3.py
├── data/
│   └── dataset/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── data2npy.py
│       └── datanpy/
│           ├── training_set.npy
│           ├── testing_set.npy
│           ├── val_set.npy
│           └── user_rating_set_all.npy
└── Model_save/
```

## Environment

The script is written for Python and PyTorch with CUDA enabled.

Required Python packages:

```bash
pip install numpy torch
```

The script sets:

```python
DEVICE = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

Use a machine with an NVIDIA GPU. To run on another GPU, change `CUDA_VISIBLE_DEVICES` in `main.py`.

## Data

The training script expects preprocessed `.npy` files under:

```text
data/Amazon_Pet_Supplies/datanpy/
```

Required files:

```text
training_set.npy
testing_set.npy
val_set.npy
user_rating_set_all.npy #consists of training set and validation set
```

## Run Training

Run the script from the `code/` directory because the script uses relative paths such as `../data/...`.

```bash
cd code
python train.py
```


## Model

`GateBoxGCN` learns two embeddings for users and items:

```text
center embedding: embed_user, embed_item
offset embedding: embed_user_dim2, embed_item_dim2
```

For each forward pass, the model performs k graph propagation layers on the user-item bipartite graph, averages embeddings from all layers, and scores user-item pairs with a box-intersection style function.

Training uses BPR-style pairwise optimization:

```text
loss = BPR loss + L2 regularization
```

Negative samples are generated each epoch through `data_utils.BPRData.ng_sample()`.

## Evaluation

Every `step` epochs, the script:

1. Saves the current model checkpoint.
2. Computes a full user-item prediction matrix.
3. Evaluates top-20 recommendation quality on the test set.


## Early Stopping
If recall does not improve for 100 epochs worth of evaluation intervals, training stops:

With the default `step = 3`, `stop1` increases by 3 after each non-improving evaluation.

## Important Notes
- Run from `code/`, not the repository root.
- The script assumes CUDA is available.
- `testing_loader_loss` and `val_loader_loss` are constructed but not used in the current training loop.
