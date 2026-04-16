# -- coding:UTF-8
import math
import time

import numpy as np
import torch


def metrics_loss(model, test_val_loader_loss, batch_size):
    start_time = time.time()
    loss_sum = []
    loss_sum2 = []
    for user, item_i, item_j in test_val_loader_loss:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        prediction_i, prediction_j, loss, loss2 = model(user, item_i, item_j)
        loss_sum.append(loss.item())
        loss_sum2.append(loss2.item())

    elapsed_time = time.time() - start_time
    test_val_loss1 = round(np.mean(loss_sum), 4)
    test_val_loss = round(np.mean(loss_sum2), 4)
    str_print_val_loss = " val loss:" + str(test_val_loss)
    _ = (prediction_i, prediction_j, elapsed_time, test_val_loss1, str_print_val_loss, batch_size)
    return test_val_loss


def precision(R, T):
    p = set(R) & set(T)
    return len(p) / len(R)


def recall(R, T):
    p = set(R) & set(T)
    if len(T) == 0:
        return 0
    return len(p) / len(T)


def hr_ndcg(indices_sort_top, index_end_i, top_k):
    hr_topK = 0
    ndcg_topK = 0

    ndcg_max = [0] * top_k
    temp_max_ndcg = 0
    for i_topK in range(top_k):
        temp_max_ndcg += 1.0 / math.log(i_topK + 2)
        ndcg_max[i_topK] = temp_max_ndcg

    max_hr = top_k
    max_ndcg = ndcg_max[top_k - 1]
    if index_end_i < top_k:
        max_hr = index_end_i * 1.0
        max_ndcg = ndcg_max[index_end_i - 1]

    count = 0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK += 1.0
            ndcg_topK += 1.0 / math.log(count + 2)
        count += 1
        if count == top_k:
            break

    hr_t = hr_topK / max_hr
    ndcg_t = ndcg_topK / max_ndcg
    prec_t = hr_topK / top_k
    recall_t = hr_topK / index_end_i
    return recall_t, ndcg_t, prec_t


def hr_ndcg_(indices_sort_top, groundTrue, top_k):
    hr_topK = 0
    ndcg_topK = 0

    ndcg_max = [0] * top_k
    temp_max_ndcg = 0
    for i_topK in range(top_k):
        temp_max_ndcg += 1.0 / math.log(i_topK + 2)
        ndcg_max[i_topK] = temp_max_ndcg

    max_hr = top_k
    max_ndcg = ndcg_max[top_k - 1]
    if len(groundTrue) < top_k:
        max_hr = len(groundTrue) * 1.0
        max_ndcg = ndcg_max[len(groundTrue) - 1]

    count = 0
    for item_id in indices_sort_top:
        if item_id in groundTrue:
            hr_topK += 1.0
            ndcg_topK += 1.0 / math.log(count + 2)
        count += 1
        if count == top_k:
            break

    if max_hr == 0:
        hr_t = 0
    else:
        hr_t = hr_topK / max_hr
    ndcg_t = ndcg_topK / max_ndcg
    prec_t = hr_topK / top_k
    return hr_t, ndcg_t, prec_t


def metrics(model, test_val_loader, top_k, num_negative_test_val, batch_size):
    HR, NDCG = [], []
    test_loss_sum = []
    _ = (num_negative_test_val, batch_size)

    test_start_time = time.time()
    for user, item_i, item_j in test_val_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j

        prediction_i, prediction_j, loss_test, loss2_test = model(user, item_i, torch.cuda.LongTensor([0]))
        test_loss_sum.append(loss2_test.item())
        elapsed_time = time.time() - test_start_time
        print("time:" + str(round(elapsed_time, 2)))

        courrent_index = 0
        courrent_user_index = 0
        for len_i, len_j in item_j:
            index_end_i = (len_i - len_j).item()
            _, indices = torch.topk(prediction_i[0][courrent_index : (courrent_index + len_i)], top_k)
            hr_t, ndcg_t = hr_ndcg(indices.tolist(), index_end_i, top_k)[:2]
            HR.append(hr_t)
            NDCG.append(ndcg_t)

            courrent_index += len_i
            courrent_user_index += 1

    test_loss = round(np.mean(test_loss_sum[:-1]), 4)
    _ = (prediction_j, loss_test, courrent_user_index)
    return test_loss, round(np.mean(HR), 4), round(np.mean(NDCG), 4)
