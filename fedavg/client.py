import torch
from sklearn.metrics import f1_score

class Client(object):

    def __init__(self, conf, model, train_dataset, val_dataset):
        """
        :param conf: 配置文件
        :param model: 全局模型
        :param train_dataset: 训练数据集
        :param val_dataset: 验证数据集
        """

        self.conf = conf

        self.local_model = model

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"],shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf["batch_size"],shuffle=True)

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.view(data.size(0),-1)
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)

                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()

            f1, acc, eval_loss = self.model_eval()
            print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_f1={3}, eval_acc={4}".format(e, loss, eval_loss, f1, acc))

        return self.local_model.state_dict()

    def model_eval(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict = []
        label = []
        for batch_id, batch in enumerate(self.val_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.local_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            predict.extend(pred.numpy())
            label.extend(target.numpy())

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        f1 = f1_score(predict, label)

        return f1, acc, total_l

