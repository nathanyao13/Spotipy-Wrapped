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
        scaler = StandardScaler()
        scaled_variables_matrix = scaler.fit_transform(self.numerical_metadata)

        count_vectorizer = CountVectorizer()
        artist_vector = count_vectorizer.fit_transform(self.df['artists'])

        item_matrix = hstack([scaled_variables_matrix,artist_vector]).tocsr()
        return item_matrix

    
        
    def recommend(self, user_likes: list):
        song_indexes = self.song_to_index[user_likes]

        liked_matrix = self.item_matrix[song_indexes]

        scores = cosine_similarity(liked_matrix, self.item_matrix).mean(axis = 0)

        scores = sorted(enumerate(scores), key=lambda i: i[1], reverse=True)

        recs = []
        i = 0
        while len(recs) < 10:
            if scores[i][0] not in song_indexes:
                recs.append(scores[i][0])
            i += 1

        return self.df.iloc[recs][['name', 'artists','album','release_date']]


if __name__ == "__main__":
    recommender_system = content_rec()      