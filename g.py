import random
import math
import os
from multiprocessing import Process, Queue

from c import SPBoard, NodeSet, MCST, SPFeatureExtractor, BaseModel, AvgModel


class DLModel(BaseModel):
    def __init__(self, num_feature, num_probability):
        super(DLModel, self).__init__(num_feature, num_probability)
        import tensorflow as tf
        num_label = 1
        num_1_1 = num_feature * 2
        num_1_2 = num_feature * 2
        num_2_1 = num_feature * 2
        num_2_2 = num_feature * 2
        num_3_1 = num_feature
        num_3_2 = num_feature
        num_4_1 = num_feature // 2
        num_4_2 = num_feature // 2
        stddev = 0.5
        x = tf.placeholder(tf.float32, [None, num_feature])
        p = tf.placeholder(tf.float32, [None, num_probability])
        y = tf.placeholder(tf.float32, [None, num_label])

        w1_1 = tf.Variable(tf.random_normal([num_feature, num_1_1], stddev=stddev))
        b1_1 = tf.Variable(tf.random_normal([num_1_1], stddev=stddev))
        w1_2 = tf.Variable(tf.random_normal([num_feature, num_1_2], stddev=stddev))
        b1_2 = tf.Variable(tf.random_normal([num_1_2], stddev=stddev))
        w2_1 = tf.Variable(tf.random_normal([num_1_1, num_2_1], stddev=stddev))
        b2_1 = tf.Variable(tf.random_normal([num_2_1], stddev=stddev))
        w2_2 = tf.Variable(tf.random_normal([num_1_2, num_2_2], stddev=stddev))
        b2_2 = tf.Variable(tf.random_normal([num_2_2], stddev=stddev))
        w3_1 = tf.Variable(tf.random_normal([num_2_1, num_3_1], stddev=stddev))
        b3_1 = tf.Variable(tf.random_normal([num_3_1], stddev=stddev))
        w3_2 = tf.Variable(tf.random_normal([num_2_2, num_3_2], stddev=stddev))
        b3_2 = tf.Variable(tf.random_normal([num_3_2], stddev=stddev))
        w4_1 = tf.Variable(tf.random_normal([num_3_1, num_4_1], stddev=stddev))
        b4_1 = tf.Variable(tf.random_normal([num_4_1], stddev=stddev))
        w4_2 = tf.Variable(tf.random_normal([num_3_2, num_4_2], stddev=stddev))
        b4_2 = tf.Variable(tf.random_normal([num_4_2], stddev=stddev))
        w5_1 = tf.Variable(tf.random_normal([num_4_1, num_probability], stddev=stddev))
        b5_1 = tf.Variable(tf.random_normal([num_probability], stddev=stddev))
        w5_2 = tf.Variable(tf.random_normal([num_4_2, num_label], stddev=stddev))
        b5_2 = tf.Variable(tf.random_normal([num_label], stddev=stddev))

        l1_1 = tf.nn.relu(tf.matmul(x, w1_1) + b1_1)
        l1_2 = tf.nn.relu(tf.matmul(x, w1_2) + b1_2)
        l2_1 = tf.nn.relu(tf.matmul(l1_1, w2_1) + b2_1)
        l2_2 = tf.nn.relu(tf.matmul(l1_2, w2_2) + b2_2)
        l3_1 = tf.nn.relu(tf.matmul(l2_1, w3_1) + b3_1)
        l3_2 = tf.nn.relu(tf.matmul(l2_2, w3_2) + b3_2)
        l4_2 = tf.nn.relu(tf.matmul(l3_2, w4_2) + b4_2)
        hypothesis_1 = tf.nn.softmax(tf.matmul(l3_1, w4_1) + b4_1)
        hypothesis_2 = tf.sigmoid(tf.matmul(l4_2, w5_2) + b5_2)

        c1 = 1e-4
        c2 = 1e-3
        cost1 = tf.reduce_mean(-p * tf.log(hypothesis_1 + 1e-8))
        cost2 = tf.reduce_mean(tf.square(y - hypothesis_2))
        regularization = c1 * tf.nn.l2_loss(w1_1)\
            + c2 * tf.nn.l2_loss(w1_2)\
            + c1 * tf.nn.l2_loss(w2_1)\
            + c2 * tf.nn.l2_loss(w2_2)\
            + c1 * tf.nn.l2_loss(w3_1)\
            + c2 * tf.nn.l2_loss(w3_2)\
            + c1 * tf.nn.l2_loss(w4_1)\
            + c2 * tf.nn.l2_loss(w4_2)\
            + c1 * tf.nn.l2_loss(w5_1)\
            + c2 * tf.nn.l2_loss(w5_2)
        cost = cost1 + cost2 + regularization

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        self.saver = saver
        self.sess = sess
        self.train = train
        self.cost = cost
        self.cost1 = cost1
        self.cost2 = cost2
        self.regularization = regularization
        self.hypothesis_1 = hypothesis_1
        self.hypothesis_2 = hypothesis_2
        self.num_label = num_label
        self.x = x
        self.p = p
        self.y = y
        return

    def fit(self, data_x, data_p, data_y):
        num_sample = len(data_x)
        print(num_sample, self.num_feature, self.num_probability, self.num_label)

        batch = num_sample // 50 + 1
        test = 0.1
        p_x = [[] for _ in range(batch)]
        p_p = [[] for _ in range(batch)]
        p_y = [[] for _ in range(batch)]
        r_x, r_p, r_y = [], [], []
        t_x, t_p, t_y = [], [], []
        for i in range(num_sample):
            p = random.random()
            if p < test:
                t_x.append(data_x[i])
                t_p.append(data_p[i])
                t_y.append(data_y[i])
            else:
                r_x.append(data_x[i])
                r_p.append(data_p[i])
                r_y.append(data_y[i])
                p = int(random.random() * batch)
                p_x[p].append(data_x[i])
                p_p[p].append(data_p[i])
                p_y[p].append(data_y[i])

        sess = self.sess
        train = self.train
        cost = self.cost
        cost1 = self.cost1
        cost2 = self.cost2
        regularization = self.regularization
        x = self.x
        p = self.p
        y = self.y

        for step in range(10001):
            if step < 9000:
                sess.run(train, feed_dict={x: p_x[step % batch], p: p_p[step % batch], y: p_y[step % batch]})
            else:
                sess.run(train, feed_dict={x: r_x, p: r_p, y: r_y})
            if step % 100 == 0:
                print(step, sess.run([cost, cost1, cost2, regularization], feed_dict={x: r_x, p: r_p, y: r_y}))
                print(step, sess.run([cost, cost1, cost2, regularization], feed_dict={x: t_x, p: t_p, y: t_y}))
                print(num_sample, self.num_feature, self.num_probability, self.num_label)
        return

    def predict(self, data_x):
        p, y = self.sess.run([self.hypothesis_1, self.hypothesis_2], feed_dict={self.x: data_x})
        return p, y

    def save_model(self, file_name):
        self.saver.save(self.sess, file_name)
        return

    def load_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


def learn_by_one_process(queue, phase, s, r, org_board, total_select):
    mcst = MCST(org_board)
    feature_extractor = SPFeatureExtractor()
    num_feature = SPFeatureExtractor.get_num_feature(org_board)
    num_probability = SPFeatureExtractor.get_num_probability(org_board)
    if phase - 1 > 0:
        model = DLModel(num_feature, num_probability)
        model.load_model("./model/tmp_{0}/{0}".format(phase - 1))
    else:
        model = AvgModel(num_feature, num_probability)
    for i in range(r):
        mcst.play(total_select, feature_extractor, model)
        data_length = feature_extractor.get_data_length()
        print("phase: {0}, process: {1}, round: {2}, sample: {3}".format(phase, s, i + 1, data_length))
    queue.put(feature_extractor)
    return


def print_kl_divergence(phase, num, feature_extractor, model):
    predict_p, predict_y = model.predict(feature_extractor.x)
    p_kl_divergence = 0.0
    for i, line_p in enumerate(predict_p):
        for j, p in enumerate(line_p):
            if feature_extractor.p[i][j] > 0.0:
                p_kl_divergence += feature_extractor.p[i][j] * (
                    math.log(feature_extractor.p[i][j] + 1e-8) - math.log(p + 1e-8)
                )
    p_kl_divergence /= len(predict_p)
    print("p_kl_divergence: ", p_kl_divergence)
    y_cross_entropy = 0.0
    y_square = 0.0
    for i, y in enumerate(predict_y):
        if feature_extractor.y[i][0] > 0.5:
            y_cross_entropy += -math.log(y[0] + 1e-8)
        else:
            y_cross_entropy += -math.log(1.0 - y[0] + 1e-8)
        y_square += (feature_extractor.y[i][0] - y[0]) ** 2
    y_cross_entropy /= len(predict_y)
    y_square /= len(predict_y)
    print("y_cross_entropy: ", y_cross_entropy)
    print("y_square: ", y_square)
    
    f = open('./model/data_{0}_{1}.txt'.format(phase, num), 'w')
    for i in range(len(predict_y)):
        print(feature_extractor.x[i], feature_extractor.p[i], feature_extractor.y[i], predict_p[i], predict_y[i], file=f)
    f.close()
    
    f = open('./model/divergence_{0}_{1}.txt'.format(phase, num), 'w')
    print("sample: ", len(predict_y), file=f)
    print("p_kl_divergence: ", p_kl_divergence, file=f)
    print("y_cross_entropy: ", y_cross_entropy, file=f)
    print("y_square: ", y_square, file=f)
    f.close()
    
    return p_kl_divergence


def train_by_one_process(phase, org_board, feature_extractor):
    num_feature = SPFeatureExtractor.get_num_feature(org_board)
    num_probability = SPFeatureExtractor.get_num_probability(org_board)
    new_model = DLModel(num_feature, num_probability)
    feature_extractor.fit(new_model)
    new_model.save_model("./model/tmp_{0}/{0}".format(phase))
    return


def compete_by_one_process(queue, phase, num, org_board, feature_extractor):
    num_feature = SPFeatureExtractor.get_num_feature(org_board)
    num_probability = SPFeatureExtractor.get_num_probability(org_board)

    if phase > 0:
        model = DLModel(num_feature, num_probability)
        model.load_model("./model/tmp_{0}/{0}".format(phase))
    else:
        model = AvgModel(num_feature, num_probability)
    
    data_length = feature_extractor.get_data_length()
    print("total_sample: {0}".format(data_length))
    p_kl_divergence = print_kl_divergence(phase, num, feature_extractor, model)
    queue.put(p_kl_divergence)
    return

def learn(phase, num_thread, r, org_board, total_select):
    print("phase: {0}, training begins".format(phase))
    queues = [Queue() for _ in range(num_thread)]
    processes = []
    for i in range(num_thread):
        process = Process(
            target=learn_by_one_process,
            args=[queues[i], phase, i + 1, r, org_board, total_select]
        )
        process.start()
        processes.append(process)

    feature_extractor = SPFeatureExtractor()
    for i in range(num_thread):
        f = queues[i].get()
        feature_extractor.x.extend(f.x.copy())
        feature_extractor.p.extend(f.p.copy())
        feature_extractor.y.extend(f.y.copy())
        processes[i].join()
        processes[i].terminate()
    
    data_length = feature_extractor.get_data_length()
    print("phase: {0}, total_sample: {1}".format(phase, data_length))
    process = Process(target=train_by_one_process, args=[phase, org_board, feature_extractor])
    process.start()
    process.join()
    process.terminate()
    
    queue = Queue()
    process = Process(target=compete_by_one_process, args=[queue, phase - 1, 1, org_board, feature_extractor])
    process.start()
    p_kl_divergence_old = queue.get()
    process.join()
    process.terminate()
    
    queue = Queue()
    process = Process(target=compete_by_one_process, args=[queue, phase, 0, org_board, feature_extractor])
    process.start()
    p_kl_divergence_new = queue.get()
    process.join()
    process.terminate()
    
    print("phase: {0}, training finished".format(phase))
    return (p_kl_divergence_old < 0.12) and (p_kl_divergence_new < 0.06)


def main():
    n = 6
    m = 2
    num_thread = 32
    
    while True:
        print("0-learn, 1-save model, 2-quit")
        try:
            key = int(input())
        except ValueError:
            continue
        if key == 0:
            total_phase = 0
            total_round = n * m
            org_board = SPBoard(n, m)
            total_select = n * m * 150
            load_model = False
            while True:
                total_phase += 1
                flag = learn(total_phase, num_thread, total_round, org_board, total_select)
                load_model = True
                if flag == False:
                    break
        if key == 1:
            name = "{0}_{1}".format(n, m)
            os.system("cp ./model/tmp_{0}/{0}.data-00000-of-00001 {1}.data-00000-of-00001".format(total_phase, name))
            os.system("cp ./model/tmp_{0}/{0}.index {1}.index".format(total_phase, name))
            os.system("cp ./model/tmp_{0}/{0}.meta {1}.meta".format(total_phase, name))
            print('Done, name: ', name)
        if key == 2:
            break

if __name__ == "__main__":
    main()
