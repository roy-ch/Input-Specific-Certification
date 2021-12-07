# evaluate a smoothed classifier on a dataset
import argparse
from time import time
from model import resnet110
from datasets import get_dataset, DATASETS, get_num_classes
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Certify with FastCertify')

# prepare data, model, output file
parser.add_argument("dataset", default='cifar10', help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")

# hyperparameters for sample size planning
parser.add_argument("--loss_type", choices=['absolute', 'relative'], help="loss type")
parser.add_argument("--max_loss", type=float, default=0.01, help="the tolerable loss of certified radius")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--max_size", type=int, default=200, help="the maximum sample size")
parser.add_argument('--n0', type=int, default=10, help='the sample size for FastCertify')
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

# hyperparameters for the input iteration
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()
print(args)


def _sample_noise(base_classifier, num_classes, x: torch.tensor, sigma: float, batchnum: int, batch_size) -> np.ndarray:
    with torch.no_grad():
        counts = np.zeros(num_classes, dtype=int)
        for _ in range(batchnum):
            batch = x.repeat((batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * sigma
            predictions = base_classifier(batch + noise).argmax(1)
            counts += _count_arr(predictions.cpu().numpy(), num_classes)
        return counts


def _count_arr(arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts


def _lower_confidence_bound(NA, N, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


def generate_iss(loss, batch_size, upper, sigma, alpha, loss_type) -> dict:
    iss = {}
    max_sample_size = upper * batch_size
    if loss_type=='absolute':
        pre=0
        for pa in list(np.arange(500 + 1) * 0.001+0.5):
            iss[pa] = upper
            opt_radius = sigma * norm.ppf(
                _lower_confidence_bound(max_sample_size * pa, max_sample_size, alpha))
            standard = opt_radius - loss
            if standard <= 0:
                iss[pa] = 0
            else:
                for num in range(pre,upper + 1):
                    sample_size = num * batch_size
                    if sigma * norm.ppf(_lower_confidence_bound(sample_size * pa, sample_size, alpha)) >= standard:
                        iss[pa] = num
                        pre=num
                        break
    if loss_type=='relative':
        for pa in list(np.arange(500 + 1) * 0.001+0.5):
            iss[pa] = upper
            opt_radius = sigma * norm.ppf(
                _lower_confidence_bound(max_sample_size * pa, max_sample_size, alpha))
            standard = opt_radius*(1- loss)
            if standard <= 0:
                iss[pa] = 0
            else:
                for num in range(upper + 1):
                    sample_size = num * batch_size
                    if sigma * norm.ppf(_lower_confidence_bound(sample_size * pa, sample_size, alpha)) >= standard:
                        iss[pa] = num
                        break
    return iss


def find_opt_batchnum(iss, pa_lower, pa_upper):
    list_p = list(iss.keys())
    pa_lower = np.clip(pa_lower, 0.0, 1.0)
    pa_upper = np.clip(pa_upper, 0.0, 1.0)
    for i, p in enumerate(list_p):
        if pa_lower <= p:
            opt_batchnum = max(iss[list_p[max(0,i - 1)]], iss[p])
            break
    for i, p in enumerate(list_p):
        if pa_upper <= p:
            opt_batchnum = max(opt_batchnum, iss[list_p[max(0,i - 1)]], iss[p])
            break
    return opt_batchnum


if __name__ == "__main__":
    t_start = time()
    batch_size = args.batch_size
    n0 = args.n0//batch_size
    alpha = args.alpha
    num_classes = 10
    sigma = args.sigma
    loss = args.max_loss
    upper = args.max_size//batch_size
    loss_type = args.loss_type


    model = resnet110().cuda()
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    if checkpoint is not None:
        print('==> Resuming from checkpoint..')
        model.load_state_dict(checkpoint['net'])
    model.eval()

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tpA\tsamplesize\tradius\tcorrect\ttime_forward", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    certify_num_total = 0
    sample_size_total = 0
    radius_total = 0
    grid = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    cnt_grid_hard =np.zeros((len(grid) + 1,), dtype=np.int)
    t1=time()
    iss=generate_iss(loss, batch_size, upper, sigma, alpha, loss_type)
    t2=time()
    time_iss=t2-t1
    for i in tqdm(range(len(dataset))):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        (x, label) = dataset[i]

        certify_num_total += 1

        t1 = time()
        # certify the prediction of g around x
        x = x.cuda()
        counts_prediction = _sample_noise(model, num_classes, x, sigma, n0, batch_size)
        # use these samples to take a guess at the top class
        prediction_uncertain = counts_prediction.argmax().item()

        pa_lower, pa_upper = proportion_confint(counts_prediction[prediction_uncertain].item(), n0 * batch_size, alpha,
                                                method="beta")

        # compute the optimal batchnum
        opt_batchnum = find_opt_batchnum(iss, pa_lower, pa_upper)
        sample_size = opt_batchnum * batch_size

        # forward
        if sample_size != 0:
            # draw more samples of f(x + epsilon)
            counts_certification = counts_prediction
            counts_certification += _sample_noise(model, num_classes, x, sigma, opt_batchnum - n0, batch_size)
            # use these samples to estimate a lower bound on pA
            nA = counts_certification[prediction_uncertain].item()
            pABar = _lower_confidence_bound(nA, sample_size, alpha)
            if pABar < 0.5:
                prediction = -1
                radius = 0
            else:
                prediction = prediction_uncertain
                radius = sigma * norm.ppf(pABar)
        else:
            pABar=pa_lower
            prediction = -1
            radius = 0
        t2 = time()

        sample_size_total += sample_size

        correct = int(prediction == label)
        if correct == 1:
            cnt_grid_hard[0] += 1
            radius_total += radius
            for j in range(len(grid)):
                if radius >= grid[j]:
                    cnt_grid_hard[j + 1] += 1

        time_forward = t2 - t1
        print(f"{i}\t{label}\t{prediction}\t{pABar:.3f}\t{sample_size}\t{radius:.3f}\t{correct}\t{time_forward:.3f}",
              file=f, flush=True)

    t_end = time()
    print(f'===Certification Summary({loss_type})===', file=f, flush=True)
    print(
        f"image number={certify_num_total}, total time={t_end - t_start}, time_iss={time_iss}, total sample size={sample_size_total}, loss control={100 * loss:.2f}%, average radius={radius_total/certify_num_total:.3f}",
        file=f, flush=True)
    print('Radius: 0.0  Number: {}  Acc: {}%'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / certify_num_total * 100),file=f, flush=True)
    for j in range(len(grid)):
        print('Radius: {}  Number: {}  Acc: {}%'.format(
            grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / certify_num_total * 100),file=f, flush=True)
    print('ACR: {}'.format(radius_total / certify_num_total), file=f, flush=True)
    f.close()

