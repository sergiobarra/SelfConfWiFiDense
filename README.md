# ML-based Stand-Alone Self-Configuration WiFI management
### Authors
* [Sergio Barrachina-Muñoz](https://github.com/sergiobarra)
* [Alessandro Chiumento](https://github.com/sergiobarra)

### Project description
We consider a fixed deployment where 6 APs and 24 STAs (or users) remain in the same position no matter the state of the system. Each AP has different STAs fixedly associated as shown in the deployment in the figure below. Notice that nodes have been spread in a way that interference is indeed critical since some cell-edge STAs are within coverage area of different APs. The idea is to find a hypothesis function `h` that given an input `x` (a state or configuration) is able to predict some performance metric `f(x)=y`, i.e., `^h(x)=y`. For that aim we rely on ML-based regressions (e.g., NNs).

<img src="https://github.com/sergiobarra/SelfConfWiFiDense/blob/master/ml_wlan/net_deployment.PNG" alt="Network deployment"
	title="Network deployment" width="400" />

### Repository description
This repository contains the Python code for evaluating the performance of different ML-models in two ways:
* General prediction error: how much different is `h` from `f`.
* Optimal prediction error: how much far is the `argmax h` from `argmax f`


### Contribute

If you want to contribute, please contact to [sergio.barrachina@upf.edu](sergio.barrachina@upf.edu)
