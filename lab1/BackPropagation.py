import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

class Layer:
    def __init__(self, InputLinks: int, OutputLinks: int, LearningRate: float = 0.1 ) -> None:
        
        self.LearningRate = LearningRate
        self.weights = np.random.normal(0,1, (InputLinks+1, OutputLinks))
        self.ForwardGradient = None
        self.BackwardGradient = None
        self.output = None
    
    def Forward (self, inputs: np.ndarray) ->np.ndarray:
        self.ForwardGradient = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        self.output = self.sigmoid(np.matmul(self.ForwardGradient, self.weights))
        return self.output
    
    def Backward (self, derivative: np.ndarray) ->np.ndarray:
        self.BackwardGradient = np.multiply(self.derivative_sigmoid(self.output), derivative)
        return np.matmul(self.BackwardGradient, self.weights[:-1].T)
    
    def Update (self) :
        Gradient = np.matmul(self.ForwardGradient.T, self.BackwardGradient)
        tmp_weights = -self.LearningRate*Gradient
        self.weights += tmp_weights

    @staticmethod
    def sigmoid (x: np.ndarray) ->np.ndarray:
        return 1.0/(1.0 + np.exp(-x))
    
    @staticmethod
    def derivative_sigmoid(x: np.ndarray) ->np.ndarray:
        return np.multiply(x, 1.0 - x)

class NN:
    def __init__(self, epoch: int  =100000, HiddenUnits: int = 4, LayerNums: int = 2, InputUnits: int = 2, LearningRate: float = 0.1) -> None:
        self.epoch = epoch
        self.HiddenUnits = HiddenUnits
        self.LayerNUms = LayerNums
        self.InputsUnits = InputUnits
        self.LearningRate = LearningRate
        self.LearningEpoch = []
        self.LearningLoss = []

        self.Layers = [Layer(InputUnits, HiddenUnits, LearningRate)]
        for i in range(LayerNums-1):
            self.Layers.append(Layer(HiddenUnits, HiddenUnits, LearningRate))
        self.Layers.append(Layer(HiddenUnits, 1, LearningRate))

    def Forward (self, inputs: np.ndarray) ->np.ndarray:
        for layer in self.Layers:
            inputs = layer.Forward(inputs)
        return inputs

    def Backward (self, derivative):
        for layer in self.Layers[::-1]:
            derivative = layer.Backward(derivative)
    
    def Update(self):
        for layer in self.Layers:
            layer.Update()
    
    def train (self, inputs: np.ndarray, labels: np.ndarray):
        for epoch in range(self.epoch):
            Pred = self.Forward(inputs)
            Loss = self.MSE(Pred=Pred, GroundTruth=labels)
            self.Backward(self.MSE_Derivative(Pred = Pred, GroundTruth=labels))
            self.Update()

            if epoch % 100 == 0:
                print(f'epoch {epoch} loss : {Loss}')
                self.LearningEpoch.append(epoch)
                self.LearningLoss.append(Loss)
            if Loss < 0.001:
                break

    def predict (self, inputs: np.ndarray) ->np.ndarray:
        Pred = self.Forward(inputs = inputs)
        print(Pred)
        return np.round(Pred)
    
    def show_result(self, x: np.ndarray, y: np.ndarray):
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.title('Ground turth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        
        pred_y = self.predict(x)
        plt.subplot(1,2,2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        
        plt.show()

    @staticmethod
    def MSE ( Pred: np.ndarray, GroundTruth: np.ndarray):
        return np.mean((Pred - GroundTruth) ** 2)
    
    @staticmethod
    def MSE_Derivative ( Pred: np.ndarray, GroundTruth: np.ndarray) ->np.ndarray:
        return 2 * (Pred - GroundTruth) / len(GroundTruth)
    
    


def main():
    # inputs, labels = generate_linear()
    inputs, labels = generate_XOR_easy()
    model = NN()
    model.train(inputs=inputs, labels=labels)
    model.show_result(x=inputs, y=labels)
    

if __name__ == '__main__':
    main()