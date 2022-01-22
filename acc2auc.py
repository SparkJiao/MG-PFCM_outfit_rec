import argparse
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--prob_file', type=str)
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()

    bs = args.batch_size
    db = 2 * bs

    pred_ls = np.load(args.pred_file)
    prob_ls = json.load(open(args.prob_file, 'r'))

    predictions = []
    tmp = []
    for idx, (pred, prob) in enumerate(zip(pred_ls, prob_ls)):
        if pred == 1:
            tmp.append([1 - prob, prob])
        else:
            tmp.append([prob, 1 - prob])

        if (idx + 1) % db == 0 or idx == len(pred_ls) - 1:
            assert len(tmp) % 2 == 0
            _half = len(tmp) // 2
            for j in range(_half):
                cmp_res = tmp[j][1] > tmp[j + _half][0]
                if cmp_res:
                    predictions.append(1)
                else:
                    predictions.append(0)
            tmp.clear()
    assert len(predictions) == len(pred_ls) // 2, (len(predictions), len(pred_ls))

    auc = sum(predictions) * 1.0 / len(predictions)
    print(auc)
