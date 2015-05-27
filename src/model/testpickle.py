import pickle


def unpickle():
    dat = pickle.load(open('testpik', 'rb'))
    return dat