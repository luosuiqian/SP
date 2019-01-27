import random
import math


def get_ucb(n, w, priors, pcut):
    ucb = []
    total = sum(n) + 1
    for i in range(len(n)):
        if n[i] == 0:
            q = 0.5
        else:
            q = w[i] / n[i]
        q += pcut * priors[i] * math.sqrt(total) / (1 + n[i])
        ucb.append((q, i))
    ucb.sort(reverse=True)
    return ucb[0][1]


def run(probas, priors):
    total_select = 1600
    p = 1.2
    pcut = 30.0
    n = [0 for _ in probas]
    w = [0 for _ in probas]
    for _ in range(total_select):
        i = get_ucb(n, w, priors, pcut)
        n[i] += 1
        if probas[i] >= random.random():
            w[i] += 1
        s = sum([v ** (1 / p) for v in n])
        new_priors = [v ** (1 / p) / s for v in n]
        print(new_priors)
    return new_priors


def main():
    probas = [0.9, 0.6, 0.5, 0.4, 0.0]
    probas = [1.0, 0.5, 0.0]
    priors = [1.0 / len(probas) for _ in probas]
    for _ in range(1):
        priors = run(probas, priors)
        print(priors)


if __name__ == "__main__":
    main()


def compete_by_model(org_board, model_a, model_b):
    win_total = 0
    lose_total = 0
    feature_extractor = SPFeatureExtractor()
    for user, model_dict in enumerate([{0: model_a, 1: model_b}, {0: model_b, 1: model_a}]):
        win = 0
        lose = 0
        for i in range(500):
            node_set = NodeSet()
            game_root = node_set.get_node(org_board)
            while game_root.winner is None:
                model = model_dict.get(game_root.board.user)
                game_root.expand(node_set, feature_extractor, model)
                move_index = game_root.play_by_probability_prior()
                game_root = game_root.edges[move_index].child_node
            if (game_root.board.user == 0 and game_root.winner == 0.0) or\
                    (game_root.board.user == 1 and game_root.winner == 1.0):
                lose += 1
            else:
                win += 1
        if user == 0:
            win_total += win
            lose_total += lose
        else:
            win_total += lose
            lose_total += win
        print("{0}-{1}".format(win, lose))
    print("Total: {0}-{1}".format(win_total, lose_total))
    return lose_total - win_total >= 20


def _get_qs_2(self, base, p):
    qs = [(edge.w / edge.n + base) ** p if edge.n > 0 else 1.0 for edge in self.edges]
    return qs


def play_by_maximum_ucb_2(self, total_select, base, p):  # Selection
    ns = self._get_ns()
    qs = self._get_qs(base, p)
    sum_n = sum(ns)
    sum_q = sum(qs)
    ucbs = [(qs[index] / sum_q * sum_n / (sum_n + total_select / 10.0) +
             edge.p * total_select / 10.0 / (sum_n + total_select / 10.0) -
             ns[index] / (sum_n + 1), index) for index, edge in enumerate(self.edges)]
    ucbs.sort(reverse=True)
    max_index = ucbs[0][1]
    return max_index


def play_by_probability_prior(self):  # Compete with model
    priors = [edge.p for edge in self.edges]
    key = random.random()
    index = 0
    while key >= priors[index]:
        key -= priors[index]
        index += 1
    return index

