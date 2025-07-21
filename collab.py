import numpy as np
import pandas as pd

class collab_rec:
    def __init__(self):
        self.R = pd.read_csv("user_item.csv", index_col = 0)
        self.R = self.R.values
        users, items = self.R.shape

        # Hyperparameters
        self.lr = 0.01
        self.reg = 0.1
        self.k = 10
        self.U = np.random.normal(scale = 1/self.k, size = (users,self.k))
        self.V = np.random.normal(scale = 1/self.k, size = (items,self.k))

        O = np.zeros((users,1)) 
        P = np.zeros((items,1)) 
        self.U = np.hstack([self.U, O, np.ones((users, 1))])
        self.V = np.hstack([self.V, np.ones((items, 1)), P])

        self.Z = np.argwhere(self.R == 1)


    def learn(self):
        max_epochs = 50
        threshold = 0.001
        prev_loss = float('inf')
        epoch = 0
        while epoch < max_epochs:
            idx = np.random.choice(len(self.Z))
            i, j = self.Z[idx]

            error = self.R[i,j] - np.dot(self.U[i], self.V[j])

            self.U[i] -= self.lr * (-error * self.V[j] + self.reg * self.U[i])
            self.V[j] -= self.lr * (-error * self.U[i] + self.reg * self.V[j])

            loss = np.square(error).mean()
            print(f"Epoch: {epoch}, Error: {error}, Loss: {loss}")
            if abs(prev_loss - loss) < threshold:
                print("Converged")
                break
            prev_loss = loss
            epoch += 1

    def user_recommendations(self, user):
        pass



if __name__ == "__main__":
    collab = collab_rec()
    collab.learn()