from c import SPBoard, NodeSet, MCST, SPFeatureExtractor, AvgModel
from g import DLModel


def compete(org_board, num_thinking, feature_extractor, model):
    print("0-first, 1-second")
    try:
        user = int(input())
    except ValueError:
        user = -1
    if user not in [0, 1]:
        print("wrong input")
        return
    print("0-normal view, 1-opposite view")
    try:
        view = int(input())
    except ValueError:
        view = -1
    if view not in [0, 1]:
        print("wrong input")
        return
    view_user = user if view == 0 else 1 - user
    mcst = MCST(org_board)
    node_set = NodeSet()
    game_root = node_set.get_node(org_board)
    while game_root.winner is None:
        game_root.board.print_board(view_user)
        if game_root.board.user != user:
            for _ in range(num_thinking):
                mcst.select(node_set, game_root, feature_extractor, model)

            ns = game_root._get_ns()
            ws = game_root._get_ws()
            qs = game_root._get_qs()
            ucbs = game_root._get_ucbs()
            ps = [edge.p for edge in game_root.edges]
            print("n:", ns)
            print("w:", ws)
            print("q:", qs)
            print("ucb:", ucbs)
            print("prior:", ps)
            print("")

            move_index = game_root.play_by_maximum_n()
            game_root = game_root.edges[move_index].child_node
        else:
            moves = game_root.board.get_avail_moves()
            print("-1 : resign")
            for i, move in enumerate(moves):
                print(str(i) + " : ", end="")
                move.print_move()
            try:
                move_index = int(input())
            except ValueError:
                move_index = -2
            if not (-1 <= move_index < len(moves)):
                print("wrong input")
                continue
            if move_index == -1:
                break
            if game_root.is_leaf():
                game_root.expand(node_set, feature_extractor, model)
            game_root = game_root.edges[move_index].child_node
            print("")
    game_root.board.print_board(view_user)
    if (game_root.board.user != user and game_root.winner == 0.0) or\
            (game_root.board.user == user and game_root.winner == 1.0):
        print("win")
    else:
        print("lose")
    return


def main():
    n = 6
    m = 2
    
    num_thinking = n * 1000
    
    org_board = SPBoard(n, m)
    num_feature = SPFeatureExtractor.get_num_feature(org_board)
    num_probability = SPFeatureExtractor.get_num_probability(org_board)
    model = AvgModel(num_feature, num_probability)

    while True:
        print("0-compete, 1-load model, 2-quit")
        try:
            key = int(input())
        except ValueError:
            continue
        if key == 0:
            feature_extractor = SPFeatureExtractor()
            compete(org_board, num_thinking, feature_extractor, model)
        if key == 1:
            try:
                print("file_name : ", end="")
                file_name = input()
                z = file_name.split('_')
                n, m = z
                n = int(n)
                m = int(m)
                org_board = SPBoard(n, m)
                num_feature = SPFeatureExtractor.get_num_feature(org_board)
                num_probability = SPFeatureExtractor.get_num_probability(org_board)
                model = DLModel(num_feature, num_probability)
                model.load_model(file_name)
            except Exception:
                print("failed to load model")
        if key == 2:
            break

if __name__ == "__main__":
    main()
