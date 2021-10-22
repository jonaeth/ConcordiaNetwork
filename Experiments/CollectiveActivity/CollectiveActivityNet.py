from Student import Student


class CollectiveActivityNet(Student):
    def __init__(self, neural_network_model, optimizer):
        super().__init__(neural_network_model, optimizer)

    def predict(self, input):
        self.model.apply(self.set_bn_eval)
        images, bboxes, bboxes_num = input
        actions_scores, activities_scores=self.model((images,bboxes[:, :, :, :-1], bboxes_num))
        return actions_scores, activities_scores

    def fit(self, loss):
        loss.backwards()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @staticmethod
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()