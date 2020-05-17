## MovieLens recommendation project ##

<img src="images/movies.jpg" width=900>  

<img src="images/recommender_systems.png" width=875>  

This project entails making movie recommendations via collaboration between user ratings and genre preferences. The dataset contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. The dataset consists of four csv files and a readme text file: links, movies, ratings, and tags with the following parameters:
* movies.csv
    * (9742, 3) rows/columns
    * 'movieId', 'title', 'genres'

* ratings.csv
    * (100836, 4) rows/columns
    * 'userId', 'movieId', 'rating', 'timestamp'

* tags.csv
    * (3683, 4) rows/columns
    * 'userId', 'movieId', 'tag', 'timestamp'

* links.csv
    * (9742, 3) rows/columns
    * 'movieId', 'imdbId', 'tmdbId'
    
* README.txt
    * much like a dictionary
    
>*The performance metric used to gauge success in this notebook is the root mean squared error. (RMSE)*
    
This data set has the propensity to recommend the most popular films(population bias), since they're the ones receiving the vast majority of ratings, some older films have litte to no ratings at all. The maximum rating is 5, lowest 1, and the average hoovers at 3.5. This may be useful in addressing cold start issues when attempting to recommend films to new users. Sort of recommending the "catch of the day". 
```python
ratings['rating'].describe()
count    100836.000000
mean          3.501557
std           1.042529
min           0.500000
25%           3.000000
50%           3.500000
75%           4.000000
max           5.000000
Name: rating, dtype: float64
```
<img src="images/ratings_long_tail.png" width= 575>  
    
**Similarity matrix's**

*Pearson similarity matrix*
```python
# corr(pearson) method adjusts for the mean by default so no further need to standardize. 
pearson_similarity = user_ratings.corr(method='pearson')
pearson_similarity.tail(5)
```
<img src="images/pearson.png" width=575>  

**User ratings based recommendations (pearson correlation matrix)**
<img src="images/pearson-math.png"> 
```python
user = [('101 Dalmatians (1996)', 1), ('2001: A Space Odyssey (1968)', 4)]

similar_movies = pd.DataFrame()
    
for movie, rating in user:
    similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index = True)
    
similar_movies.sum().sort_values(ascending=False).head(10)

2001: A Space Odyssey (1968)                                                   1.333735
Blade Runner (1982)                                                            0.822194
Aliens (1986)                                                                  0.639186
Clockwork Orange, A (1971)                                                     0.614576
Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)    0.603032
Apocalypse Now (1979)                                                          0.586503
Full Metal Jacket (1987)                                                       0.577457
Platoon (1986)                                                                 0.549155
Dark City (1998)                                                               0.541112
Alien (1979)                                                                   0.537291
```
*Cosine similarity matrix*
```python
# to compute a similarity score three options are available: euclidean, correlation (pearson), and cosine
tfV = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfV_matrix = tfV.fit_transform(movies['genres'])

cosine_similarity = linear_kernel(tfV_matrix, tfV_matrix)
```
**Genre based recommendations**
<img src="images/cosine.png"> 
```python
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def genre_based_recommendations(title):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]

    return titles.iloc[movie_indices]

genre_based_recommendations('Zombieland (2009)')

7154                                    Zombieland (2009)
8434                                    Zombeavers (2014)
8565                    Dead Snow 2: Red vs. Dead (2014) 
9035         Scouts Guide to the Zombie Apocalypse (2015)
62                             From Dusk Till Dawn (1996)
6251                             Snakes on a Plane (2006)
6324                                         Feast (2005)
11                     Dracula: Dead and Loving It (1995)
654     Tales from the Crypt Presents: Bordello of Blo...
1478                                      Gremlins (1984)
```
#### Future work:
Deeper study into neural network recommendation systems through pytorch via Nvidia cuda (gpu processing) and tinkering with the various loss functions: mean absolute error, mean square error, smooth L1 loss, negative log-likeihood loss, cross-entropy loss, kullback-leibler divergence, marging ranking loss, hinge embedding loss, and cosine embedding loss. 