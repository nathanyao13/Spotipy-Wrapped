import pandas as pd

# Class for searching songs in the dataset
class search:    
    def __init__(self):
        self.df = pd.read_csv("metadata.csv")

    # Input: Song and Artist
    def search_by_song_and_artist(self, song, artist):
        filt = self.df['name'].str.contains(song, case=False, na=False) & self.df['artists'].str.contains(artist, case=False, na=False)
        matches = self.df[filt]
        
        if len(matches) == 0:
            print(f"No songs found matching '{song}' by '{artist}'")
            return None
            
        return matches[['name', 'artists', 'album', 'id']]
    
    # Input: Song
    def search_by_song(self, song):
        filt = self.df['name'].str.contains(song, case=False, na=False)
        matches = self.df[filt]
        
        if len(matches) == 0:
            print(f"No songs found matching '{song}'")
            return None
            
        return matches[['name', 'artists', 'album', 'id']]
    
    # Input Artist
    def search_by_artist(self, artist):
        filt = self.df['artists'].str.contains(artist, case=False, na=False)
        matches = self.df[filt]
        
        if len(matches) == 0:
            print(f"No artists found matching '{artist}'")
            return None
            
        return matches[['name', 'artists', 'album', 'id']]