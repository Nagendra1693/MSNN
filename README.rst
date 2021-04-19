Multilayer Extreme Learning Machine with Subnetwork nodes
--------

Extreme Learining Machine is a single layer feed forward network. According to the paper mentioned below, we can replace the hidden node with subwork formed by several hidden nodes which naturally form s biological learning.

There are few differences between the proposed model and the conventional ELM:

* Hidden nodes are generated randomly one by one in ELM. Subnetwork nodes are used in the proposed model instead of single hidden node.
* Number of hidden nodes and diemension if outputs in ELM are independent. In the proposed model, dimension of general nodes and outputs are also independent but number of hidden nodes in a general neuron should be equal to the dinmension of the outputs.

Highlights:
    - Efficient matrix math implementation
    - GPU acceleration is included.
    - This is extremely fast when compared to other deeplearing autoencoders.

Main classes:
    - MSNN for implementing Multilayer ELM with subnetworks
    - ELM for the classification of data.

Example usage::
    >>> m = MSNN()
    >>> m.fit(x = x_train_m, y = y_train_m, d = 41, l_max = 2, act_fn = 'sigmoid', coefficient = 20)
    >>> x_train_new = m.transform(x_train_m)
    >>> x_test_new = m.transform(x_test_m)
    >>> 
    >>> mm_scaler = MinMaxScaler()
    >>> x_train_norm = mm_scaler.fit_transform(x_train_new) 
    >>> x_test_norm = mm_scaler.transform(x_test_new)
    >>> elm = ELM(hiddennodes = 1000, activation = 'relu')
    >>> train_acc = elm.fit(x_train_norm, ytrain)
    >>> test_acc = elm.evaluate(x_test_norm, y_test)
    >>> print('\tTraining Accuracy : ', train_acc)
    >>> print('\tTesting Accuracy : ', test_acc)

References:
	- Guang-Bin Huang, Qin-Yu Zhu and Chee-Kheong Siew, "Extreme learning machine: a new learning scheme of feedforward neural networks," 2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No.04CH37541), Budapest, Hungary, 2004, pp. 985-990 vol.2, doi: 10.1109/IJCNN.2004.1380068.
	- Yang Y, Wu QM. Multilayer Extreme Learning Machine With Subnetwork Nodes for Representation Learning. IEEE Trans Cybern. 2016 Nov;46(11):2570-2583. doi: 10.1109/TCYB.2015.2481713. Epub 2015 Oct 9. PMID: 26462250.
	- https://personal.ntu.edu.sg/egbhuang/elm_codes.html
	- https://documen.tician.de/pycuda/index.html
	- https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.inv.html
	- https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
