import json,os
import pandas as pd
from conf import conf
import torch
from fedavg.server import Server
from fedavg.client import Client
from fedavg.datasets import MyTabularDataset
from fedavg.models import MLP

def get_tabular_data():
    """
    :return: 加载数据
    """
    ###训练数据路径
    train_dataset_file = conf["train_dataset"]
    #测试数据路径
    test_dataset_file = conf["test_dataset"]

    train_datasets = {}
    val_datasets = {}
    ##各节点数据量
    number_samples = {}

    ##读取数据集,训练数据拆分成训练集和测试集
    for key in train_dataset_file.keys():
        train_dataset = pd.read_csv(train_dataset_file[key])

        val_dataset = train_dataset[:int(len(train_dataset)*conf["split_ratio"])]
        train_dataset = train_dataset[int(len(train_dataset)*conf["split_ratio"]):]
        train_datasets[key] = MyTabularDataset(train_dataset, conf["label_column"])
        val_datasets[key] = MyTabularDataset(val_dataset,conf["label_column"])

        number_samples[key] = len(train_dataset)

    ##测试集,在Server端测试模型效果
    test_dataset = pd.read_csv(test_dataset_file)

    #模型输入维度
    n_input = test_dataset.shape[1] - 1
    test_dataset = MyTabularDataset(test_dataset,conf["label_column"])
    print("数据加载完成!")

    return train_datasets, val_datasets, test_dataset, n_input


if __name__ == '__main__':

    train_datasets, val_datasets, test_dataset, n_input= get_tabular_data()

    ###初始化每个节点聚合权值
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("聚合权值初始化")

    ##保存节点
    clients = {}
    # 保存节点模型
    clients_models = {}

    if conf["model_name"] == 'mlp':
        model = MLP(n_input, 512, conf["num_class"])

    server = Server(conf, model, test_dataset)

    print("Server初始化完成!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key])

    print("参与方初始化完成！")

    for e in range(conf["global_epochs"]):

        for key in clients.keys():
            print('training {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            clients_models[key] = model_k

        #联邦聚合
        server.model_aggregate(clients_models, client_weight)

        f1, acc, loss = server.model_eval()

        print("Epoch %d, global_f1: %f, global_acc: %f, global_loss: %f\n" % (e, f1, acc, loss))

    #保存模型
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"],conf["model_file"]))

    print("联邦训练完成，模型保存在{0}目录下!".format(conf["model_dir"]))