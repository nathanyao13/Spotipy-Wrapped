# Music Recommender Systems: Spotipy Wrapped

My music taste has evolved over the years, from Maroon 5 to Playboi Carti to Mac Miller. Along the way, I discovered new songs from social media, friends, and Spotify’s built‑in recommendations like AI DJ and the "Made for You" tab. This project explores how systems like Spotify recommend music, and demonstrates two core approaches from recommender systems: collaborative filtering and content‑based filtering. The goal is to connect the theory to a concrete, working implementation with Spotify data. <br>


<img src="https://i.imgur.com/HmOCA03.png" width="500">

---

## Content-Filtering
The logic behind my content-based recommendation system was to compute similarity scores between the submitted liked songs and the songs in the dataset. <br>

<b> Data Preperation: </b> <br>
Before that, similarity scores can only be calculated with numerical values, so I changed the artists feature to numerical using the Count Vectorizer. <br>
<br>
The Count Vectorizer vectorizes the artists based on how popular they are. I chose to use this vectorizer because if two songs share the same artist, they should be considered more similar. Other choices such as the TF-IDF Vectorizer penalizes popular artists and downweights them. <br>
<br>
Similarly, I also standardized each of the numerical values onto the same scale so what does not weigh more than the other. 

<b>Similarity Scores:</b> <br>
For similarity scores, I used cosine similarity which vectorizes each song's metadata and compares the angle between the vectors of two songs.<br>

The intuition is that if the angle is smaller, the more similar the songs are, and vice versa. <br>

Taking the cosine of the angle between two vectors will give a value between -1 and 1. However, all of the features I use have values that are non-negative which results in values from 0 to 1, in practice. The closer the value is to 1, the more similar the vectors/songs are to each other.

When recommending songs, the recommender system calculates the mean of the similarity scores of a given list of liked songs, sorts the song data by similarity scores, and return the top 10 songs bases on similarity. 


---
## Collaborative-Filtering

The algorithm behind my collaboration-filtering system is a latent-factor method, I learned in school called Matrix Factorization. <br>

The intuition behind this technique is to decompose a sparse matrix, such as user liked songs, into the product of two matrices with one that corresponds to users and another to items. <br>

The algorithm used learns the rank of the two matrices that minimizes the mean squared error between the sparse user matrix and the two low-dimension matrices. <br>

<b>Learning Algorithm:</b> <br>

<img src="https://i.imgur.com/ppcTZNT.png" width="500">

<b>Recommending:</b> <br>
Given a user index, the algorithm grabs the user's row from the user matrix and multiplies it with the transpose of the items matrix to predict scores for all songs. <br>

After filtering out the songs the user has already liked, the algorithm sorts the remaining songs by their predicted scores and returns the top 10 highest-scoring recommendations.

---


## Further Exploration

### Hybrid Recommender Systems

- 

### Dataset Limitations

- 

---

Sources:

    Datasets:
    - Spotify API data: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download
    - User data was generated using a python script located in eda.ipynb


    Research and Coding:
    - https://www.cs.cmu.edu/~mgormley/courses/10601/
    - https://www.stratascratch.com/blog/step-by-step-guide-to-building-content-based-filtering/#contentbased_filtering_implementation 
