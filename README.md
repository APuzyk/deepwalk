# Deepwalk

This crate provides a implementation of [Deepwalk](https://arxiv.org/pdf/1403.6652.pdf) [Perozzi, Al-Rfou, Skiena 2014].

To the run  you need to provide a whitespace separated edgelist of a graph where the node ids are integers.  See karate_netwrok.txt for an example.

#### How To

The binary takes one arguments, a json file that contains paramter information.

```
deepwalk config.json
```

In config.json you must provide

* learning_rate: float - The starting point for the learning rate.  This is decremented linearly with each iteration.
* vector_dim: usize - The dimensionality desired for the final output vectors.
* walk_length: usize - The length of the random walk taken for each node
* window_size: usize - The window size applied to the walk.  Note nodes +-window_size are used (e.g. a window size of 2 give the two nodes before and 2 nodes after the target node as part of the window)
* num_iterations: usize - The number of iterations to run the algorithm for.  Note we have not implemented early stopping at this time.
* input_file: string -  The edge list file described above.
* perf_file: string -  A file location to write the performance information to (iteration learning_rate error) for each iteration
* weight_file: string - A file location to write the final weights/vectors
* nthreads: usize - The number of threads to use for running the algorithm.  If 0 or 1 is selected this will run single threaded

#### Karate Example

We have included an example run of the algorithm on the karate network included in this directory.

The configurations for this run can be found in `karate_config.json`.  We have plotted the output weights along with the canonical communities.

![Karate](https://github.com/APuzyk/deepwalk/blob/master/karate_2d.png)

We observed similar results to what Perozzi, et.al saw in their runs on the Karate network.  Communities were moved into similar posotions in the 2d space of the weight matrix.
