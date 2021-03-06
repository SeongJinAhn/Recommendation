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
<img src="/image/MAP.PNG" width="30%" height="30%">
<img src="/image/MLE.PNG" width="30%" height="30%">

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
## [Bayesian Personalized Ranking from Implicit Feedback (BPR)](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)  


## [Matrix Factorization Techniques for Recommender Systems (LFM)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)  
Use p_i, q_j which are latent representations of users & items
Predict by 'dot of p_i, q_j'
![MAP](image/LFM.PNG)  
## [Neural Collaborative Filtering (NCF)](https://www.ijcai.org/proceedings/2017/0447.pdf)  
Automatically learn the function F
## [Deep Matrix Factorization Models for Recommender Systems (DeepMF)](https://www.ijcai.org/proceedings/2017/0447.pdf)
Use LFM (use inner product to calculate the interaction between users & items)  
Use representation learning (Deep Structured Semantic Models) instead of learning F.

## [Group recommendation using feature space representing behavioral tendency and power balance among members (NCF)](https://dl.acm.org/citation.cfm?id=2043953)  


## [Genarative Model for Group Recommendation (COM)](https://dl.acm.org/ft_gateway.cfm?ftid=1494840&id=2623616)  


## [Attentive Group Recommendation (AGREE)](https://www.ijcai.org/proceedings/2017/0447.pdf)  
U : users {u_1, u_2, ... u_n} G : groups {g_1, g_2, ... g_s} V : items {v_1, v_2, ... v_m}  
Y : group-item interaction [y_ij]s x m  
M : user-item interaction [r_ij]n x m  

g_l(j) : 논문에서 group의 특성을 잘 담았다고 정의한 vector  
        \sigma ( a(j,t) * u_t) + q_l

<img src="/image/AGREE0.PNG">

특정 그룹(고정)내에서 t를 구매하는데에 j의 영향력을 구하고 그것들의 linear sum을 해당그룹이 t를 구매할 가능성으로 보았다.  
α(j,t) : t가 j를 사게하는데의 영향력  
$\sigma$ {t} ( a(j,t) * u_t ) : linear sum of a(j,t) * Embedding(j)으로 영향력이 반영된 group의 vector  
q_l : 구성원의 합이 아닌, 그룹 자체만의 특성을 나타낸 vector  

<img src="/image/AGREE1.PNG" width="80%" height="80%">
<img src="/image/AGREE2.PNG" width="80%" height="80%">  

## [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617)  
Learn the importance of each feature interaction from data via a neural network.  
Purpose : Learn the importance of each feature interaction  
Common Solution1 : Augment a feature vector with product of features.(Poloynomial Regression : PR)  
Problem of CS1: In sparse dataset, only a few cross features are observed. So the parameters for unobserved cross features cannot be estimated.  
Common Solution2 : Learning an embedding vector for each feature, and estimate the weight for any cross feature (Factorization Matrix : FM)  
Problem of CS2 : May result in suboptimal prediction (local maxima)
Solution they suggest : Attentive Facorization Matrix
 

<img src="/image/attentive_network.PNG" width="80%" height="80%">

## Group Preference Based Bayesian Personalized Ranking for One-Class Collaborative Filtering (GBPR)

## [A3NCF : An adaptive aspective attention model for rating prediction](https://www.ijcai.org/proceedings/2018/0521.pdf)  
![A3NCF](image/A3NCF_architecture.PNG)  
  
#### 1. Topic Model  
Latent Dirichlet Allocation
<img src="/image/A3NCF_topic.PNG" width="50%" height="50%">  
#### 2. Embedding  
#### 3. Fusion  
fuse the embedded features and review-based features for better representation learning.  
adopt the **addiction fusion (Not Concat)** which has been applied in RBLT and ITLFM.  
#### 4. Attentive Interaction  
<img src="/image/A3NCF_attention.PNG" width="50%" height="50%">  
<img src="/image/A3NCF_attention2.PNG" width="30%" height="30%">  
#### 5. Rating Prediction  
<img src="/image/A3NCF_prediction.PNG" width="50%" height="50%">  

## HoP-Rec
1. User U Item의 Adjacent Matrix를 기반으로 Graph를 만들고, 각 user에서 pagerank를 돌려서 neighbor item을 선택  
2. Personalized Ranking에서 다음 neighbor들을 비교한다
<img src="/image/HoP-Rec_1.PNG" width="50%" height="50%">  
<img src="/image/HoP-Rec_2.PNG" width="30%" height="30%">  

## PinSAGE
Random Walk기반으로 Neighbor선택이 아니라, Neighbor들의 가중치를 선택  
GraphSAGE의 Aggregate function을 그 가중치를 바탕으로 형성

## Decagon
Multi-relation을 예측하는 방법  
embedding된 vector에 trainable한 diagonal matrix와, relation에 따라 다른 trainable한 matrix를 곱하는 방식으로 예측

## IntentGC
bit-wise => vector-wise  

## Vectorized Relational Graph Convolutional Network(VR-GCN)
기존 GCN이 multi-relation을 잘 예측x =>  VR-GCN  
directed graph를 가정했다. (우리 문제와는 다르지만, idea를 얻어보자)  
u를 예측하고 싶을때 u-(r_i)->v_i면 v_i - r_i  
			 u<-(r_j)-v_j면 v_j + r_i

## ClusterGCN
By expanding until the cluster in which the node is inhibit excessive expansion by neighbors
<img src="/image/clusterGCN_1.PNG" width="50%" height="50%">  
