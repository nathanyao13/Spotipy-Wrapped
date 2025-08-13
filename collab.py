# Collaborative Filtering Implementation

import numpy as np
import pandas as pd

class collab_rec:
    def __init__(self):
        self.songs_df = pd.read_csv("metadata.csv")
        self.R = pd.read_csv("user_item.csv", index_col = 0)
        self.R = self.R.values
        users, items = self.R.shape

        # Hyperparameters
        self.lr = 0.01
        self.reg = 0.001
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
            indices = np.random.permutation(len(self.Z))

            # stochastic gradient descent
            for idx in indices:
                i, j = self.Z[idx]

                error = self.R[i,j] - np.dot(self.U[i], self.V[j])

                u_copy = self.U[i].copy()

                self.U[i] -= self.lr * (-error * self.V[j] + self.reg * self.U[i])
                self.V[j] -= self.lr * (-error * u_copy + self.reg * self.V[j])

            
            # calculate error from dot product between columns and the sparse matrix
            square_error_sum = 0
            for i, j in self.Z:
                prediction = np.dot(self.U[i], self.V[j])
                square_error_sum += (self.R[i, j] - prediction) ** 2
            
            # update mse, regularization term, and loss
            mse = 0.5 * square_error_sum / len(self.Z)
            reg_term = 0.5 * self.reg * (np.sum(self.U**2) + np.sum(self.V**2))
            loss = mse + reg_term

            print(f"Epoch: {epoch}, MSE: {mse:.4f}, Loss: {loss:.4f}")

            # end early due to converging 
            if abs(prev_loss - loss) < threshold:
                print("Converged")
                break

            prev_loss = loss
            epoch += 1


    def user_recommendations(self, user):
        # vector-matrix multiplication
        scores = self.U[user] @ self.V.T

        # remove liked songs
        liked_items = np.where(self.R[user] == 1)[0]
        scores[liked_items] = -np.inf

        # grab top ten songs
        top_items = np.argsort(scores)[::-1][:10]
        return self.songs_df.iloc[top_items][['name', 'artists', 'album', 'release_date']]

if __name__ == "__main__":
    collab = collab_rec()
    collab.learn()