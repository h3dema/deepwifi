# Temporal Convolutional Networks

They are a variation of Convolutional Neural Network architecture for sequence modelling tasks.
The main idea is to convolve only on the elements from current timestamp or earlier in the previous layer (no “leakage” from future to past, also called causal convolutions) using a 1D fully-convolutional network architecture (FCN).
In the FCN, each hidden layer is the same length as the input layer, and zero padding of length (kernel size − 1) is added to keep subsequent layers the same length as previous ones.


# The code

This code is based on the github implementation called [Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN)](https://github.com/locuslab/TCN/) that uses PyTorch.




```
from tcn import compiled_tcn

# generates a Keras model
model = compiled_tcn(...)

model.fit(x_train, y_train)

model.predict(x_test)
```


# References


* Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

* Lea, C., Vidal, R., Reiter, A., & Hager, G. D. (2016, October). Temporal convolutional networks: A unified approach to action segmentation. In European Conference on Computer Vision(pp. 47–54). Springer, Cham.

* Kalchbrenner, N., Espeholt, L., Simonyan, K., Oord, A. V. D., Graves, A., & Kavukcuoglu, K. (2016). Neural machine translation in linear time. arXiv preprint arXiv:1610.10099.

## Blogs

* [TEMPORAL CONVOLUTIONAL NETWORKS](https://medium.com/@raushan2807/temporal-convolutional-networks-bfea16e6d7d2)