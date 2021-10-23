from Concordia.Student import Student


class CollectiveActivityNet(Student):
    def __init__(self, neural_network_model, optimizer):
        super().__init__(neural_network_model, optimizer)

    # TODO Can we implement this more gerenally as ClassificationStudent?
    def predict(self, input):
        self.model.apply(self.set_bn_eval)
        images, bboxes, bboxes_num = input
        actions_scores, activities_scores = self.model((images, bboxes[:, :, :, :-1], bboxes_num))
        return actions_scores, activities_scores

    def fit(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # TODO do we need it? Does it need to be here?
    @staticmethod
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

