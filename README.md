# Music Recommender Systems: Spotipy Wrapped

My music taste has evolved over the years, from Maroon 5 to Playboi Carti to Mac Miller. Along the way, I discovered new songs from social media, friends, and Spotify’s built‑in recommendations like AI DJ and the "Made for You" tab. This project explores how systems like Spotify recommend music, and demonstrates two core approaches from recommender systems: collaborative filtering and content‑based filtering. The goal is to connect the theory to a concrete, working implementation with Spotify data. <br>


<img src="https://i.imgur.com/HmOCA03.png" width="500">

---

## Content-Based Filtering  

The logic behind my content-based recommendation system is to compute similarity scores between a user’s liked songs and all songs in the dataset.  

**Data Preparation**  
Similarity scores require numerical features, so I converted the `artists` column into numerical form using **CountVectorizer**. I chose CountVectorizer because if two songs share the same artist, they should be considered more similar. In contrast, **TF-IDF Vectorizer** would penalize popular artists by downweighting them.  

In addition, I standardized all numerical features to the same scale to ensure no single feature disproportionately influenced similarity.  

**Similarity Scores**  
I used **cosine similarity** to compare the metadata vectors of songs. The idea is simple:  
- Smaller angles between vectors mean higher similarity.  
- Cosine similarity returns values between -1 and 1, but since all features are non-negative, values range from 0 to 1 in practice.  
- A score closer to 1 means the songs are more similar.  

To generate recommendations, the system:  
1. Calculates the average similarity score for each song against all liked songs.  
2. Sorts songs by similarity score.  
3. Returns the top 10 most similar songs.  

---

## Collaborative Filtering  

For collaborative filtering, I implemented a **latent factor model** using **Matrix Factorization**, a method I learned in school.  

The idea is to decompose a sparse **user–item interaction matrix** (e.g., liked songs) into the product of two lower-dimensional matrices:  
- One representing **user preferences**.  
- One representing **item (song) characteristics**.  

The model learns these matrices by minimizing the **mean squared error** between the original sparse matrix and the product of the two low-rank matrices.  

**Learning Algorithm**  

<img src="https://i.imgur.com/ppcTZNT.png" width="500">  

**Recommending**  
1. Select the user’s row from the **user matrix**.  
2. Multiply it with the **transpose of the item matrix** to predict scores for all songs.  
3. Filter out songs the user already liked.  
4. Sort the remaining songs by predicted score.  
5. Return the top 10 highest-scoring recommendations.  

---


## Further Exploration

### Hybrid Recommender Systems
Both collaborative filtering and content‑based filtering have trade‑offs. Collaborative filtering works well without explicit metadata but struggles with new items or users (cold start). In contrast, content‑based filtering can make recommendations out of the box if metadata is available, but may miss patterns in collective behavior. Industry systems often combine methods—for example, blending scores from collaborative filtering and content‑based filtering or dynamically switching strategies depending on data availability.

### Dataset Limitations
For any Machine Learning or AI model, data quality is critical. Unfortunately, Spotify has removed many API endpoints that previously provided rich song metadata and user interaction data. As a result, this project relied on an outdated Spotify dataset and a sparse user-item matrix generated with a Python script. While these resources allowed model implementation and testing, they did not fully capture the true performance of the recommendation system. With more recent and comprehensive data, the model’s predictive accuracy and recommendation quality would improve substantially, offering a more reliable assessment of its real-world potential.

---

Sources:

    Datasets:
    - Spotify API data: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download
    - User data was generated using a Python script located in eda.ipynb


    Research and Coding:
    - https://www.cs.cmu.edu/~mgormley/courses/10601/
    - https://www.stratascratch.com/blog/step-by-step-guide-to-building-content-based-filtering/#contentbased_filtering_implementation 
