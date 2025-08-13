import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

class content_rec:
    def __init__(self):
        self.df = pd.read_csv("metadata.csv")
        self.numerical_metadata = self.preprocess()
        self.item_matrix = self.create_item_matrix()
        self.song_to_index = pd.Series(self.df.index, index=self.df['id']).drop_duplicates()
        
        

    def preprocess(self):
        df = self.df.copy()
        df.drop(columns = ['id', 'name', 'album', 'album_id', 'artists', 'artist_ids',
                           'track_number', 'disc_number','key','mode','time_signature','release_date'], inplace = True)
        df['explicit'] = df['explicit'].astype(int)

        return df
    
    def create_item_matrix(self):
        # standardize numerical features
        scaler = StandardScaler()
        scaled_variables_matrix = scaler.fit_transform(self.numerical_metadata)

        # vectorize artist names to numerical values
        # already standardized
        count_vectorizer = CountVectorizer()
        artist_vector = count_vectorizer.fit_transform(self.df['artists'])

        item_matrix = hstack([scaled_variables_matrix,artist_vector]).tocsr()
        return item_matrix

    
        
    def recommend(self, user_likes: list):
        song_indexes = self.song_to_index[user_likes]

        liked_matrix = self.item_matrix[song_indexes]

        # similarity between liked songs and item matrix
        scores = cosine_similarity(liked_matrix, self.item_matrix).mean(axis = 0)

        # sort scores
        scores = sorted(enumerate(scores), key=lambda i: i[1], reverse=True)

        # grab 10 that are not in the user likes list
        recs = []
        i = 0
        while len(recs) < 10:
            if scores[i][0] not in song_indexes:
                recs.append(scores[i][0])
            i += 1

        return self.df.iloc[recs][['name', 'artists','album','release_date']]


if __name__ == "__main__":
    recommender_system = content_rec()      