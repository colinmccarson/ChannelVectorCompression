import numpy as np
import torch

'''data = np.load('./data/training_data_new.npz')
lst = data.files
i = 0
for item in lst:
    print(item)
    print(data[item].shape, i)'''
class HGenerator:
    def __init__(self, data_dir):
        self.__packed = np.load(data_dir)
        self.A = self.__packed[self.__packed.files[0]]
        self.C = self.__packed[self.__packed.files[1]]
        self.H = self.__gen_H(self.A, self.C)

    def __gen_H(self, A, C):
        theta_matrix = torch.rand((A.shape[0], C.shape[0]))
        imaginary_unit = 1j
        expd = torch.exp(imaginary_unit*theta_matrix)
        e_tensor = torch.diag_embed(expd)  # should have shape K x L X L
        H = torch.matmul(torch.matmul(A.unsqueeze(1), e_tensor), C.unsqueeze(0))  # TODO check
        return torch.stack((H.real, H.imag))

    def make_train_test_split(self, dir, test_proportion=.1):
        testbound = test_proportion * self.H.shape[0]
        length = self.H.shape[0]
        test = self.H.index_select(0, torch.tensor([0, testbound]))
        train = self.H.index_select(0, torch.tensor([testbound, length]))
        torch.save(test, dir + "/test.pt")
        torch.save(train, dir + "/train.pt")

