import numpy as np
# from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from random import shuffle
import pickle


def hadamard(x, y):
    return x*y

def l1_weight(x, y):
    return np.absolute(x-y)

def l2_weight(x, y):
    return np.square(x-y)

def concate(x, y):
    return np.concatenate((x, y), axis=1)

def average(x, y):
    return (x+y)/2


def load_model(path):
    with open(path, 'rb') as f:
        emb_vertex = pickle.load(f)
        sign_w = pickle.load(f)
        proj_w = pickle.load(f)
        id2vertex = pickle.load(f)
        vertex2id = pickle.load(f)
        edge_source_id = pickle.load(f)
        edge_target_id = pickle.load(f)
        edge_sign = pickle.load(f)
    return emb_vertex, sign_w, proj_w, id2vertex, vertex2id, edge_source_id, edge_target_id, edge_sign


def construct_dataset(edge_sign, id2vertex):
    pos_edges = []
    neg_edges = []
    fake_edges = []
    for edge, sign in edge_sign.items():
        if sign == 1:
            pos_edges.append(edge)
        else:
            neg_edges.append(edge)
    shuffle(pos_edges)
    n = len(neg_edges)
    sub_pos_edges = pos_edges[:n]
    n_nodes = len(id2vertex)
    n_fake = 0
    while True:
        edge = np.random.choice(n_nodes, 2)
        if (edge[0], edge[1]) not in edge_sign:
            fake_edges.append((edge[0], edge[1]))
            n_fake += 1
        if n_fake == n:
            break
    x = sub_pos_edges + neg_edges + fake_edges
    y = [1]*len(sub_pos_edges) + [-1]*len(neg_edges) + [0]*len(fake_edges)
    return x, y


def link_prediction(x, y, emb_vertex, emb_context, op):
    kf = KFold(len(x), n_folds=10)
    s_idx, t_idx, signs = [], [], []
    for (s, t), y in zip(x ,y):
        s_idx.append(s)
        t_idx.append(t)
        signs.append(y)
    s_emb = emb_vertex[s_idx]
    t_emb = emb_context[t_idx]
    signs = np.asarray(signs)
    kf_accu = []
    for i, (train_index, test_index) in enumerate(kf):
        y_train, y_test = signs[train_index], signs[test_index]
        s_train, s_test = s_emb[train_index], s_emb[test_index]
        t_train, t_test = t_emb[train_index], t_emb[test_index]
        x_train = op(s_train, t_train)
        x_test = op(s_test, t_test)
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(x_train, y_train)
        test_preds = clf.predict(x_test)
        accuracy = np.mean(test_preds == y_test)
        print("Folder:%i, Accuracy: %f" % (i, accuracy))
        kf_accu.append(accuracy)
    print("Average Accuracy: %f" % (np.mean(kf_accu)))

#emb_vertex, sign_w, proj_w, id2vertex, vertex2id, edge_source_id, edge_target_id, edge_sign = \
    load_model('lbl_wiki_edit_emb.pkl')

#link_x, link_y = construct_dataset(edge_sign=edge_sign, id2vertex=id2vertex)
#link_prediction(x=link_x, y=link_y, emb_vertex=emb_vertex, emb_context=proj_w, op=hadamard)








