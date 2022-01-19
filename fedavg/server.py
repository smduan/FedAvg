import torch
from sklearn.metrics import f1_score

class Server(object):

    def __init__(self, conf, model, test_dataset):

        self.conf = conf

        self.global_model = model

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf["batch_size"],shuffle=True)

    def model_aggregate(self, clients_model, weights):

        new_model = {}

        for name, params in self.global_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        for key in clients_model.keys():

            for name, param in clients_model[key].items():
                new_model[name]= new_model[name] + clients_model[key][name] * weights[key]

        self.global_model.load_state_dict(new_model)

    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict = []
        label = []
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

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






