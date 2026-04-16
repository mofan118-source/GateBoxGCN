# -- coding:UTF-8
import os
import random
import time
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_utils
import evaluate_v3 as evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DEVICE = "cuda"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)


def readD(set_matrix, num_):
    degrees = []
    for i in range(num_):
        degree = len(set_matrix.get(i, set()))
        degrees.append(0.0 if degree == 0 else 1.0 / degree)
    return degrees


def readTrainSparseMatrix(set_matrix, user_num, item_num, u_d, i_d, is_user):
    indices = []
    values = []
    if is_user:
        d_i, d_j = u_d, i_d
        n_row, n_col = user_num, item_num
    else:
        d_i, d_j = i_d, u_d
        n_row, n_col = item_num, user_num

    for i in set_matrix:
        for j in set_matrix[i]:
            indices.append([i, j])
            values.append(np.sqrt(d_i[i] * d_j[j]))

    indices = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    values = torch.tensor(values, dtype=torch.float32, device=DEVICE)
    return torch.sparse_coo_tensor(
        indices.t(),
        values,
        size=(n_row, n_col),
        dtype=torch.float32,
        device=DEVICE,
    ).coalesce()


def readTrainSparseMatrix_dim2(set_matrix, user_num, item_num, u_d, i_d, is_user):
    indices = []
    values = []
    if is_user:
        d_i = u_d
        n_row, n_col = user_num, item_num
    else:
        d_i = i_d
        n_row, n_col = item_num, user_num

    for i in set_matrix:
        for j in set_matrix[i]:
            indices.append([i, j])
            values.append(d_i[i])

    indices = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    values = torch.tensor(values, dtype=torch.float32, device=DEVICE)
    return torch.sparse_coo_tensor(
        indices.t(),
        values,
        size=(n_row, n_col),
        dtype=torch.float32,
        device=DEVICE,
    ).coalesce()


def expand_degree_tensor(degrees, factor_num):
    degree_tensor = torch.tensor([[value] for value in degrees], dtype=torch.float32, device=DEVICE)
    return degree_tensor.expand(-1, factor_num)


def average_tensors(tensors):
    return sum(tensors) / len(tensors)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_eval_message(elapsed_time, recall, ndcg, precision):
    return (
        f"time:{round(elapsed_time, 2)}\t test"
        f" recall:{recall} ndcg:{ndcg} prec:{precision}"
    )


class GateBoxGCN(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        factor_num,
        user_item_matrix,
        item_user_matrix,
        user_item_matrix_dim2,
        item_user_matrix_dim2,
        d_i_train,
        d_j_train,
    ):
        super(GateBoxGCN, self).__init__()
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.user_item_matrix_dim2 = user_item_matrix_dim2
        self.item_user_matrix_dim2 = item_user_matrix_dim2

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.embed_user_dim2 = nn.Embedding(user_num, factor_num)
        self.embed_item_dim2 = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_user_dim2.weight, std=0.01)
        nn.init.normal_(self.embed_item_dim2.weight, std=0.01)

        self.d_i_train = d_i_train
        self.d_j_train = d_j_train

    def box_intersection_batch(self, c_u, o_u, c_i, o_i, is_pos=True):
        l_ui = torch.max(c_u - o_u, c_i - o_i)
        u_ui = torch.min(c_u + o_u, c_i + o_i)
        s = u_ui - l_ui

        g_hard = ((o_u >= 0) & (o_i >= 0)).float()
        score_pos = (g_hard * s).sum(dim=-1)

        g_soft = torch.sigmoid(o_u) * torch.sigmoid(o_i)
        mask_neg = ((o_u < 0) | (o_i < 0)).float()

        eps = 1e-3
        o_u_eps = torch.where(o_u >= 0, o_u, o_u.new_full(o_u.shape, eps))
        o_i_eps = torch.where(o_i >= 0, o_i, o_i.new_full(o_i.shape, eps))
        l_cf = torch.max(c_u - o_u_eps, c_i - o_i_eps)
        u_cf = torch.min(c_u + o_u_eps, c_i + o_i_eps)
        beneficial = (u_cf - l_cf > 0).float().detach()

        w = 2.0 * beneficial - 1.0
        gate_grad = (mask_neg * g_soft) * w
        grad_only = (gate_grad - gate_grad.detach()).sum(dim=-1)
        return score_pos + grad_only

    def iou_batch(self, c_u, o_u, c_i, o_i, is_pos=True):
        return self.box_intersection_batch(c_u, o_u, c_i, o_i, is_pos)

    def _update_offset(
        self,
        num_nodes,
        center_self,
        center_neigh,
        o_prev,
        src_idx,
        dst_idx,
        gamma=0.6,
        eta=4.0,
        clamp_min=0.0,
    ):
        del o_prev, gamma, eta, clamp_min
        device = center_self.device
        n, d = num_nodes, center_self.size(1)

        diff = (center_neigh[src_idx] - center_self[dst_idx]).abs()
        sum_abs = torch.zeros((n, d), device=device, dtype=diff.dtype).index_add(0, dst_idx, diff)
        cnt = torch.bincount(dst_idx, minlength=n).to(sum_abs.dtype).view(n, 1)
        mask = cnt > 0
        safe_cnt = torch.where(mask, cnt, torch.ones_like(cnt))
        d_env = (sum_abs / safe_cnt) * mask

        ones = torch.ones((src_idx.numel(), 1), device=device)
        cnt = torch.zeros((n, 1), device=device).index_add(0, dst_idx, ones).clamp_min_(1.0)
        sum_ = torch.zeros((n, d), device=device).index_add(0, dst_idx, center_neigh[src_idx])
        sumsq = torch.zeros((n, d), device=device).index_add(0, dst_idx, center_neigh[src_idx] ** 2)
        mean = sum_ / cnt
        var = (sumsq / cnt) - (mean * mean)

        gate = torch.sigmoid(var)
        #gate = var
        return gate * d_env

    def propagation(
        self,
        items_embedding,
        users_embedding,
        items_embedding_dim2,
        users_embedding_dim2,
        gamma=0.6,
        eta=4.0,
        clamp_min=0.0,
    ):
        cU_k = torch.sparse.mm(self.user_item_matrix, items_embedding)
        cI_k = torch.sparse.mm(self.item_user_matrix, users_embedding)

        ui = self.user_item_matrix.coalesce()
        u_idx, i_idx = ui.indices()

        nU, _ = cU_k.shape
        nI, _ = cI_k.shape

        oU_k = self._update_offset(
            nU,
            cU_k,
            cI_k,
            users_embedding_dim2,
            src_idx=i_idx,
            dst_idx=u_idx,
            gamma=gamma,
            eta=eta,
            clamp_min=clamp_min,
        )
        oI_k = self._update_offset(
            nI,
            cI_k,
            cU_k,
            items_embedding_dim2,
            src_idx=u_idx,
            dst_idx=i_idx,
            gamma=gamma,
            eta=eta,
            clamp_min=clamp_min,
        )
        return cU_k, cI_k, oU_k, oI_k

    def forward(self, user, item_i, item_j, training=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        users_embedding_dim2 = self.embed_user_dim2.weight
        items_embedding_dim2 = self.embed_item_dim2.weight

        user_layers = [users_embedding]
        item_layers = [items_embedding]
        user_dim2_layers = [users_embedding_dim2]
        item_dim2_layers = [items_embedding_dim2]

        cur_users = users_embedding
        cur_items = items_embedding
        cur_users_dim2 = users_embedding_dim2
        cur_items_dim2 = items_embedding_dim2
        for _ in range(3):
            cur_users, cur_items, cur_users_dim2, cur_items_dim2 = self.propagation(
                cur_items,
                cur_users,
                cur_items_dim2,
                cur_users_dim2,
            )
            user_layers.append(cur_users)
            item_layers.append(cur_items)
            user_dim2_layers.append(cur_users_dim2)
            item_dim2_layers.append(cur_items_dim2)

        gcn_users_embedding = average_tensors(user_layers)
        gcn_items_embedding = average_tensors(item_layers)
        gcn_users_embedding_dim2 = average_tensors(user_dim2_layers)
        gcn_items_embedding_dim2 = average_tensors(item_dim2_layers)

        if not training:
            return gcn_users_embedding, gcn_users_embedding_dim2, gcn_items_embedding, gcn_items_embedding_dim2, 0, 0

        user_dim1 = F.embedding(user, gcn_users_embedding)
        user_dim2 = F.embedding(user, gcn_users_embedding_dim2)
        item_i_dim1 = F.embedding(item_i, gcn_items_embedding)
        item_i_dim2 = F.embedding(item_i, gcn_items_embedding_dim2)
        item_j_dim1 = F.embedding(item_j, gcn_items_embedding)
        item_j_dim2 = F.embedding(item_j, gcn_items_embedding_dim2)

        volumn_similarity_i = self.iou_batch(user_dim1, user_dim2, item_i_dim1, item_i_dim2, is_pos=True)
        volumn_similarity_j = self.iou_batch(user_dim1, user_dim2, item_j_dim1, item_j_dim2, is_pos=False)

        l2_regulization = 0.001 * (
            user_dim1 ** 2
            + item_i_dim1 ** 2
            + item_j_dim1 ** 2
            + user_dim2 ** 2
            + item_i_dim2 ** 2
            + item_j_dim2 ** 2
        ).sum(dim=-1)
        loss2 = -((volumn_similarity_i - volumn_similarity_j).sigmoid().log().mean())
        loss = loss2 + l2_regulization.mean()
        return gcn_users_embedding, gcn_users_embedding_dim2, gcn_items_embedding, gcn_items_embedding_dim2, loss, loss2


def largest_indices(ary, n):
    flat = ary.ravel()
    n = int(n)
    n = min(n, flat.size)
    if n <= 0:
        return tuple(np.array([], dtype=int) for _ in ary.shape)

    cand = np.argpartition(flat, -n)[-n:]
    scores = flat[cand]
    order = np.lexsort((cand, -scores))
    top = cand[order]
    return np.unravel_index(top, ary.shape)


def make_pred(model, user_num, item_num, batch_size=1):
    scores_matrix = np.zeros((user_num, item_num))
    test_users = torch.arange(user_num, device=DEVICE)
    test_items = torch.arange(item_num, device=DEVICE)

    with torch.no_grad():
        gcn_users_embedding, gcn_users_embedding_dim2, gcn_items_embedding, gcn_items_embedding_dim2, _, _ = model(
            None, None, None, False
        )

        user_dim1 = F.embedding(test_users, gcn_users_embedding)
        user_dim2 = F.embedding(test_users, gcn_users_embedding_dim2)

        for i in range(0, item_num, batch_size):
            items_batch = test_items[i : i + batch_size]
            item_dim1 = F.embedding(items_batch, gcn_items_embedding)
            item_dim2 = F.embedding(items_batch, gcn_items_embedding_dim2)
            similarity = model.iou_batch(
                user_dim1.unsqueeze(1),
                user_dim2.unsqueeze(1),
                item_dim1.unsqueeze(0),
                item_dim2.unsqueeze(0),
            )
            scores_matrix[:, i : i + batch_size] = similarity.cpu().numpy()

    return scores_matrix


def test_evaluation(pred_matrix, training_user_set, testing_user_set, item_num, top_k):
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    all_items = set(range(item_num))

    test_start_time = time.time()
    for user_id in testing_user_set:
        pos_items = list(testing_user_set[user_id])
        pos_count = len(pos_items)
        
        neg_items = list(all_items - training_user_set[user_id] - testing_user_set[user_id])
        candidate_items = pos_items + neg_items
        pred_scores = pred_matrix[user_id][candidate_items]
        ranked_indices = list(largest_indices(pred_scores, top_k)[0])
        recall_t, ndcg_t, precision_t = evaluate.hr_ndcg(ranked_indices, pos_count, top_k)

        recall_scores.append(recall_t)
        ndcg_scores.append(ndcg_t)
        precision_scores.append(precision_t)

    elapsed_time = time.time() - test_start_time
    recall = round(np.mean(recall_scores), 4)
    ndcg = round(np.mean(ndcg_scores), 4)
    precision = round(np.mean(precision_scores), 4)

    eval_message = format_eval_message(elapsed_time, recall, ndcg, precision)
    print(eval_message)
    return recall, precision, ndcg


def build_dataloaders(
    training_user_set,
    testing_user_set,
    val_user_set,
    item_num,
    training_set_count,
    testing_set_count,
    val_set_count,
    user_rating_set_all,
    batch_size,
):
    train_dataset = data_utils.BPRData(
        train_dict=training_user_set,
        num_item=item_num,
        num_ng=5,
        is_training=True,
        data_set_count=training_set_count,
        all_rating=user_rating_set_all,
    )
    testing_dataset_loss = data_utils.BPRData(
        train_dict=testing_user_set,
        num_item=item_num,
        num_ng=5,
        is_training=True,
        data_set_count=testing_set_count,
        all_rating=user_rating_set_all,
    )
    val_dataset_loss = data_utils.BPRData(
        train_dict=val_user_set,
        num_item=item_num,
        num_ng=5,
        is_training=True,
        data_set_count=val_set_count,
        all_rating=user_rating_set_all,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testing_loader_loss = DataLoader(testing_dataset_loss, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader_loss = DataLoader(val_dataset_loss, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, testing_loader_loss, val_loader_loss


def main():
    dataset_base_path = "../data/Amazon_Pet_Supplies"
    dataset = "Amazon_Pet_Supplies"
    user_num = 8324
    item_num = 12130
    factor_num = 256
    batch_size = 1024 * 2
    top_k = 20
    step = 3
    run_id = "stest"

    set_seed(SEED)
    print(run_id)

    path_save_base = f"./log/{dataset}/newloss{run_id}"
    path_save_model_base = f"../Model_save/{dataset}/s{run_id}"
    ensure_dir(path_save_base)
    ensure_dir(path_save_model_base)

    script_path = Path(__file__).resolve()
    copyfile(script_path, Path(path_save_base) / f"{script_path.stem}{run_id}.py")

    training_user_set, training_item_set, training_set_count = np.load(
        dataset_base_path + "/datanpy/training_set.npy",
        allow_pickle=True,
    )
    testing_user_set, testing_item_set, testing_set_count = np.load(
        dataset_base_path + "/datanpy/testing_set.npy",
        allow_pickle=True,
    )
    val_user_set, val_item_set, val_set_count = np.load(
        dataset_base_path + "/datanpy/val_set.npy",
        allow_pickle=True,
    )
    user_rating_set_all = np.load(
        dataset_base_path + "/datanpy/user_rating_set_all.npy",
        allow_pickle=True,
    ).item()
    _ = (testing_item_set, val_item_set)

    u_d = readD(training_user_set, user_num)
    i_d = readD(training_item_set, item_num)

    sparse_u_i = readTrainSparseMatrix(training_user_set, user_num, item_num, u_d, i_d, True)
    sparse_i_u = sparse_u_i.t().coalesce()
    sparse_u_i_dim2 = readTrainSparseMatrix_dim2(training_user_set, user_num, item_num, u_d, i_d, True)
    sparse_i_u_dim2 = readTrainSparseMatrix_dim2(training_item_set, user_num, item_num, u_d, i_d, False)

    d_i_train = expand_degree_tensor(u_d, factor_num)
    d_j_train = expand_degree_tensor(i_d, factor_num)

    train_loader, testing_loader_loss, val_loader_loss = build_dataloaders(
        training_user_set,
        testing_user_set,
        val_user_set,
        item_num,
        training_set_count,
        testing_set_count,
        val_set_count,
        user_rating_set_all,
        batch_size,
    )
    _ = (testing_loader_loss, val_loader_loss)

    model = GateBoxGCN(
        user_num,
        item_num,
        factor_num,
        sparse_u_i,
        sparse_i_u,
        sparse_u_i_dim2,
        sparse_i_u_dim2,
        d_i_train,
        d_j_train,
    ).to(DEVICE)
    optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("--------training processing-------")
    smallest_loss = 1000
    stop1 = 0
    stop2 = 0
    best_recall_for_stop = 0

    result_path = Path(path_save_base) / "results.txt"
    with open(result_path, "w+") as result_file:
        for epoch in range(1000):
            neg_u = neg_i = tot_u = tot_i = 0
            epoch_seed = SEED + epoch
            #set_seed(epoch_seed)

            model.train()
            train_loader.dataset.ng_sample()
            print("train data of ng_sample is  end")
            start_time = time.time()

            train_loss_sum = []
            train_loss_sum2 = []
            for user, item_i, item_j in train_loader:
                user = user.to(DEVICE)
                item_i = item_i.to(DEVICE)
                item_j = item_j.to(DEVICE)

                optimizer_bpr.zero_grad(set_to_none=True)
                (
                    gcn_users_embedding,
                    gcn_users_embedding_dim2,
                    gcn_items_embedding,
                    gcn_items_embedding_dim2,
                    loss,
                    loss2,
                ) = model(user, item_i, item_j, True)
                loss.backward()
                optimizer_bpr.step()

                train_loss_sum.append(loss.item())
                train_loss_sum2.append(loss2.item())
                with torch.no_grad():
                    neg_u += (gcn_users_embedding_dim2 < 0).sum().item()
                    tot_u += gcn_users_embedding_dim2.numel()
                    neg_i += (gcn_items_embedding_dim2 < 0).sum().item()
                    tot_i += gcn_items_embedding_dim2.numel()

            cur_u_pct = 100 * neg_u / max(tot_u, 1)
            cur_i_pct = 100 * neg_i / max(tot_i, 1)
            print(f"epoch {epoch}: users<0={cur_u_pct:.6f}% | items<0={cur_i_pct:.6f}%")

            elapsed_time = time.time() - start_time
            train_loss = round(np.mean(train_loss_sum[:-1]), 4)
            train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
            train_message = (
                f"epoch:{epoch} time:{round(elapsed_time, 1)}\t train loss:{train_loss}={train_loss2}+"
            )
            print("--train--", elapsed_time)
            print(train_message)

            if epoch % step == 0:
                model_path = Path(path_save_model_base) / f"epoch{epoch}.pt"
                torch.save(model.state_dict(), model_path)
                model.eval()

                result_file.write(train_message)
                result_file.write("\n")
                result_file.flush()

                pred_matrix = make_pred(model, user_num, item_num)
                recall, precision, ndcg = test_evaluation(
                    pred_matrix,
                    training_user_set,
                    testing_user_set,
                    item_num,
                    top_k,
                )
                if recall > best_recall_for_stop:
                    best_recall_for_stop = recall
                    stop1 = 0
                else:
                    stop1 += step

            if train_loss < smallest_loss:
                smallest_loss = train_loss
                stop2 = 0
            else:
                stop2 += 1
            print(smallest_loss, stop1, stop2)

            if stop1 >= 100:
                break


if __name__ == "__main__":
    main()
