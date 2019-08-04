# Recommendation
![MAP](image/recommend_formula.PNG)  
estimating the ratings of each unobserverd entry in Y, which are used for ranking the items.


## Explicit Feedback vs Implicit Feedback
### Explicit Feedback
Netflix : star rating for movies   
Facebook : like/un-like   
Youtube : thumbs-up/down buttons   

There are a lot of cases that, explicit feedback do not exist.  
It makes hard to predict.  

### Implicit Feedback
reflect opinion through observing user behavior  
purchase history, browsing history, search patterns, or even mouse movements  

Implicit Feedback do not perfectly reflect preference.
e.g) buying present
  
## Similarity
### Cosine(x,y)
### Pearson Correlation Coefficient
### Spearman Rank

## MAP & MLE
![MAP](image/MAP.PNG)
![MLE](image/MLE.PNG)

## Loss Function
### Squared Loss
based on the assumption that observation are generated from a Gaussian distribution
### Cross-Entropy Loss
target value is a binary classification problem

## Past Approaches
### User-based CF
need Neighbor that is n users who are resemble with the user we want to predict  
[Herlocker, Jonathan L., et al. "An algorithmic framework for performing collaborative filtering." 22nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 1999. Association for Computing Machinery, Inc, 1999.](https://experts.umn.edu/en/publications/an-algorithmic-framework-for-performing-collaborative-filtering)
### Item-based CF
just recommend the item resemble with selected

# Paper
## Bayesian Personalized Ranking from Implicit Feedback (BPR)


## Matrix Factorization Techniques for Recommender Systems (LFM)
Use p_i, q_j which are latent representations of users & items
Predict by 'dot of p_i, q_j'
![MAP](image/LFM.PNG)  
## Neural Collaborative Filtering (NCF)
Automatically learn the function F
## Deep Matrix Factorization Models for Recommender Systems (DeepMF)
Use LFM (use inner product to calculate the interaction between users & items)  
Use representation learning (Deep Structured Semantic Models) instead of learning F.

# Group Recommendation
how to aggregate the preference of group members to decide a group's choice on item.

## Group recommendation using feature space representing behavioral tendency and power balance among members (NCF)


## Genarative Model for Group Recommendation (COM)  


## Attentive Group Recommendation (AGREE)
U : users {u_1, u_2, ... u_n} G : groups {g_1, g_2, ... g_s} V : items {v_1, v_2, ... v_m}  
Y : group-item interaction [y_ij]s x m M : user-item interaction [r_ij]n x m  
