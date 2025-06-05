

def valid_model(model, X, y):
    ''' valid the model on dataset '''
    correct, total = 0, 0
    predict = model.predict(X)
    total += y.size
    correct += (predict == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def extract_data(data_loader):
    ''' extract data from pytorch loader and return numpy list '''
    X, Y = [], []  # 
    for x, y in data_loader:
        X = x.numpy()
        Y = y.numpy()
    return X, Y 


