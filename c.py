import random
import math


class Board(object):
    def __init__(self):
        return

    def get_hash(self):
        raise NotImplementedError()

    def get_winner(self):  # 1:win, 0:lost, None:unfinished
        raise NotImplementedError()

    def check_by_special_rule(self):  # 1:lost, 0:None
        raise NotImplementedError()

    def get_avail_moves(self):  # list of moves
        raise NotImplementedError()

    def take_action(self, move):  # next board
        raise NotImplementedError()

    def print_board(self, user):  # 0:first view, 1:second view
        raise NotImplementedError()


class Move(object):
    def __init__(self):
        return

    def print_move(self):
        raise NotImplementedError()


class FeatureExtractor(object):
    def __init__(self):
        return

    def get_num_feature(self, board):
        raise NotImplementedError()
    
    def get_num_probability(self, board):
        raise NotImplementedError()
    
    def add_data(self, board, moves, probabilities, value):
        raise NotImplementedError()

    def get_data_length(self):
        raise NotImplementedError()

    def fit(self, model):
        raise NotImplementedError()

    def predict(self, model, board, moves):
        raise NotImplementedError()


class BaseModel(object):
    def __init__(self, num_feature, num_probability):
        self.num_feature = num_feature
        self.num_probability = num_probability
        return

    def fit(self, data_x, data_p, data_y):
        raise NotImplementedError()

    def predict(self, data_x):
        raise NotImplementedError()

    def save_model(self, file_name):
        raise NotImplementedError()

    def load_model(self, file_name):
        raise NotImplementedError()


class NodeSet(object):
    def __init__(self):
        self.hash_key_node = {}

    def get_node(self, board):
        key = board.get_hash()
        if key in self.hash_key_node:
            return self.hash_key_node[key]
        node = Node(board)
        self.hash_key_node[key] = node
        return node


class Node(object):
    def __init__(self, board):
        self.board = board
        self.edges = None
        self.winner = board.get_winner()
        return

    def is_leaf(self):
        if self.winner is not None:
            return True
        if self.edges is None:
            return True
        return False

    def expand(self, node_set, feature_extractor, model):
        if self.winner is not None:
            return self.winner
        moves = self.board.get_avail_moves()
        probabilities, value = feature_extractor.predict(model, self.board, moves)
        self.edges = []
        for index, move in enumerate(moves):
            edge = Edge(node_set, self.board, move, probabilities[index])
            self.edges.append(edge)
        return value

    def _get_ns(self):
        ns = [edge.n for edge in self.edges]
        return ns

    def _get_ws(self):
        ws = [edge.w for edge in self.edges]
        return ws

    def _get_qs(self):
        qs = [edge.w / edge.n if edge.n > 0 else 1.0 for edge in self.edges]
        return qs

    def get_probabilities(self):
        ns = self._get_ns()
        s = sum(ns)
        return [v / s for v in ns]

    def _get_ucbs(self):
        ns = self._get_ns()
        qs = self._get_qs()
        sum_n_root = math.sqrt(sum(ns) + 1)
        ucbs = [(edge.s, qs[i] + edge.p * sum_n_root / (1 + ns[i]), i) for i, edge in enumerate(self.edges)]
        return ucbs
    
    def play_by_maximum_ucb(self):  # Selection
        ucbs = self._get_ucbs()
        ucbs.sort(reverse=True)
        max_index = ucbs[0][2]
        return max_index

    def play_by_probability_n(self):  # Play
        ns = self._get_ns()
        key = random.random() * sum(ns)
        index = 0
        while key >= ns[index]:
            key -= ns[index]
            index += 1
        return index

    def play_by_maximum_n(self):  # Compete
        ns = self._get_ns()
        avail_ns = [(n, index) for index, n in enumerate(ns)]
        avail_ns.sort(reverse=True)
        move_index = avail_ns[0][1]
        return move_index


class Edge(object):
    def __init__(self, node_set, board, move, probability):
        self.move = move
        child_board = board.take_action(move)
        self.child_node = node_set.get_node(child_board)
        self.n = 0
        self.w = 0.0
        self.p = probability
        self.s = child_board.check_by_special_rule()
        return

    def add_value(self, v):
        self.n += 1
        self.w += v
        return


class MCST(object):
    def __init__(self, org_board):
        self.org_board = org_board
        return

    def select(self, node_set, root, feature_extractor, model):
        if root.is_leaf():
            value = root.expand(node_set, feature_extractor, model)
            return value
        max_index = root.play_by_maximum_ucb()
        child = root.edges[max_index].child_node
        value = self.select(node_set, child, feature_extractor, model)
        value = 1.0 - value
        root.edges[max_index].add_value(value)
        return value

    def _play(self, node_set, root, total_select, feature_extractor, model):
        for _ in range(total_select):
            self.select(node_set, root, feature_extractor, model)
        if root.is_leaf():
            winner = root.winner
        else:
            index = root.play_by_probability_n()
            child_node = root.edges[index].child_node
            winner = 1.0 - self._play(node_set, child_node, total_select, feature_extractor, model)
            moves = [edge.move for edge in root.edges]
            probabilities = root.get_probabilities()
            feature_extractor.add_data(root.board, moves, probabilities, winner)
        return winner

    def play(self, total_select, feature_extractor, model):
        node_set = NodeSet()
        root = node_set.get_node(self.org_board)
        self._play(node_set, root, total_select, feature_extractor, model)
        return


class SPBoard(Board):
    def __init__(self, n, m, user=None, up=None, down=None):
        super(SPBoard, self).__init__()
        if user is None:
            user = 0
        if up is None:
            up = [m] * n + [0] * (n + 1)
        if down is None:
            down = [m] * n + [0] * (n + 1)
        self.n = n
        self.m = m
        self.user = user
        self.up = up
        self.down = down
        self.winner = self._check_winner()
        return

    def _check_winner(self):
        if self.user == 0 and (sum(self.up) == 0 or self.down[2 * self.n] > 0):
            return 0.0
        if self.user == 1 and (sum(self.down) == 0 or self.up[2 * self.n] > 0):
            return 0.0
        return None

    def get_winner(self):
        return self.winner

    def check_by_special_rule(self):
        if self.user == 0:
            my, rival = self.up, self.down
        else:
            my, rival = self.down, self.up
        pos = 1
        while True:
            if my[pos] < self.m:
                return 0
            if my[self.n * 2 - pos] > 0:
                return 0
            if rival[self.n * 2 - pos] > 0:
                return 1
            pos += 1
        return 0

    def get_avail_moves(self):
        moves = []
        if self.user == 0:
            arr = self.up
        else:
            arr = self.down
        for pos, w in enumerate(arr):
            for cnt in range(1, w + 1):
                if arr[pos + 1] + cnt <= self.m:
                    moves.append(SPMove(pos, cnt))
        return moves

    def take_action(self, move):
        up = self.up[:]
        down = self.down[:]
        p, w = move.pos, move.cnt
        if self.user == 0:
            my, rival = up, down
        else:
            my, rival = down, up
        my[p] -= w
        my[p + 1] += w
        if rival[self.n * 2 - p - 1] == w:
            rival[self.n * 2 - p - 1] = 0
        child = SPBoard(self.n, self.m, 1 - self.user, up, down)
        return child

    def print_board(self, user):
        if user == 0:
            up = self.down
            down = self.up
        else:
            up = self.up
            down = self.down
        for j in range(2 * self.n + 1):
            print("{0:02d} ".format(j), end="")
        print("")
        print("")
        for i in range(self.m, 0, -1):
            for _ in range(2):
                for j in range(2 * self.n + 1):
                    if up[j] >= i:
                        print("** ", end="")
                    else:
                        print("   ", end="")
                print("")
            print("")
        print("-" * (6 * self.n + 2) + " " + ('v' if self.user == user else '^'))
        print("")
        for i in range(1, self.m + 1):
            for _ in range(2):
                for j in range(2 * self.n, -1, -1):
                    if down[j] >= i:
                        print("** ", end="")
                    else:
                        print("   ", end="")
                print("")
            print("")
        for j in range(2 * self.n, -1, -1):
            print("{0:02d} ".format(j), end="")
        print("")
        print("")
        return

    def get_hash(self):
        s = self.user
        for i in self.down:
            s = s * (self.m + 1) + i
        for i in self.up:
            s = s * (self.m + 1) + i
        return s


class SPMove(Move):
    def __init__(self, pos, cnt):
        super(SPMove, self).__init__()
        self.pos = pos
        self.cnt = cnt
        return

    def print_move(self):
        print("({0:02d} {1})".format(self.pos, self.cnt))
        return


class SPFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(SPFeatureExtractor, self).__init__()
        self.x = []
        self.p = []
        self.y = []
        return

    @classmethod
    def get_num_feature(cls, board):
        return board.n * board.m * 4
    
    @classmethod
    def get_num_probability(cls, board):
        return board.n * board.m * 2

    @classmethod
    def get_feature(cls, board):
        if board.user == 0:
            my, rival = board.up, board.down
        else:
            my, rival = board.down, board.up
        features = []
        for j in range(board.n * 2):
            for i in range(1, board.m + 1):
                features.append(1 if my[j] >= i else 0)
        for j in range(board.n * 2):
            for i in range(1, board.m + 1):
                features.append(1 if rival[j] >= i else 0)
        return features

    @classmethod
    def get_label(cls, board, moves, probabilities, value):
        s = sum(probabilities)
        labels = [0.0] * cls.get_num_probability(board)
        for i, move in enumerate(moves):
            index = move.pos * board.m + move.cnt - 1
            labels[index] = float(probabilities[i]) / s
        values = [value]
        return labels, values

    @classmethod
    def get_probabilities_and_value(cls, board, moves, p, y):
        probabilities = []
        for i, move in enumerate(moves):
            index = move.pos * board.m + move.cnt - 1
            probabilities.append(p[index] + 0.1)
        s = sum(probabilities)
        probabilities = [p / s for p in probabilities]
        value = y[0]
        return probabilities, value

    def add_data(self, board, moves, probabilities, value):
        x = self.get_feature(board)
        p, y = self.get_label(board, moves, probabilities, value)
        self.x.append(x)
        self.p.append(p)
        self.y.append(y)
        return

    def get_data_length(self):
        return len(self.x)

    def fit(self, model):
        model.fit(self.x, self.p, self.y)
        return

    @classmethod
    def predict(cls, model, board, moves):
        x = SPFeatureExtractor.get_feature(board)
        p, y = model.predict([x])
        probabilities, value = cls.get_probabilities_and_value(board, moves, p[0], y[0])
        return probabilities, value


class AvgModel(BaseModel):
    def __init__(self, num_feature, num_probability):
        super(AvgModel, self).__init__(num_feature, num_probability)
        return

    def fit(self, data_x, data_p, data_y):
        pass

    def predict(self, data_x):
        p = [1.0 / self.num_probability for _ in range(self.num_probability)]
        y = [0.5]
        return [p for _ in range(len(data_x))], [y for _ in range(len(data_x))]

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
