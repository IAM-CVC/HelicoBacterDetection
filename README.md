# HelicoBacterDetection
Detection of HPylori in immunohistochemically stained histological images using Autoencoders.

We propose to use autoencoders to learn the latent patterns of healthy
patches and, under the assumption that such autoencoder will poorly reconstruct
the red channel associated to H. pylori staining, we formulate a specific measure
of this reconstruction error in HSV space. ROC analysis is used to set the optimal
threshold on this measure, as well as, the percentage of positive patches in a
sample that determines the presence of H. pylori. 

We provide Python code and a sample of the database. The full dataset with annotations (around 19 GB) will be able to be downloaded soon here

