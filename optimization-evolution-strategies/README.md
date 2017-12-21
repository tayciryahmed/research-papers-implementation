# (1,λ)ES with AII adaptation scheme for Black-Box Optimization Benchmarking for Noiseless Functions (bbob suite)

### Description

This is an implementation of AII algorithm by N. Hansen, A. Ostermeier, and A. Gawelczyk from "On the Adaptation of Arbitrary Normal Mutation Distributions in Evolution Strategies: The Generating Set Adaptation". In Proceedings of the Sixth International Conference on Genetic Algorithms, pp. 57-64, 1995. ([Paper](https://www.lri.fr/~hansen/GSAES.pdf))

### How to test this implementation 

This algorithm is run on the bbob test suite from [COCO](https://github.com/numbbo/coco). For requirements and guidelines about using COCO platform, refer to the Github repo or their website. Note that we used python to work with this platform. 

To run the code, you need to install the COCO platform ([installation guide](https://github.com/numbbo/coco)).

Then, you can run the optimization problems by typing :

``` bash
$ python A2.py bbob 20
```

To generate the post-processing results, you can use:

``` bash
$ python -m cocopp -o path_to_results exdata
```