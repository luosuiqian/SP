import random
import math


def get_ucb(n, w, priors, total_select, p, base):
    ucb = []
    q = [(w[i] / n[i] + base) ** p if n[i] > 0 else 1.0 for i in range(len(n))]
    total_n = sum(n) + 1
    total_q = sum(q)
    for i in range(len(n)):
        u = (total_n / (total_select + total_n) * q[i] / total_q
             + total_select / (total_select + total_n) * priors[i]
             - n[i] / (total_n + 1))
        ucb.append((u, i))
    ucb.sort(reverse=True)
    return ucb[0][1]


def run(probas, priors, flag=False):
    total_select = 1600
    p = 0.001
    base = 0.02
    n = [0 for _ in probas]
    w = [0 for _ in probas]
    for _ in range(total_select):
        i = get_ucb(n, w, priors, total_select, p, base)
        n[i] += 1
        if probas[i] >= random.random():
            w[i] += 1
        if flag:
            s = sum([v for v in n])
            new_priors = [v / s for v in n]
            print(new_priors)
    s = sum([v for v in n])
    new_priors = [v / s for v in n]
    return new_priors


def main():
    probas = [0.9, 0.6, 0.5, 0.4, 0.3]
    # probas = [1.0, 0.5, 0.0]
    priors = [1.0 / len(probas) for _ in probas]
    for _ in range(100):
        priors = run(probas, priors)
        print(priors)
    print("")
    priors = run(probas, priors, True)


if __name__ == "__main__":
    main()

