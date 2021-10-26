# Evolution Prediction of Topics with Graph AutoEncoders (GAE)

We have performed Top2Vec on a set of documents and created a set of topics with their weighted vectors. We use this data
and create a graph using Weighted Jaccard Similarity. Afterward, we transform this graph to Pytorch Geometric DataFrame
and apply GAE to perform link prediction between topics. 
