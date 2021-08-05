# Cilk STL Algorithms
This repo contains an early version of an implementation of the C++ parallel STL algorithms written using [OpenCilk](https://cilk.mit.edu/).

# Usage notes
The main functions are intended to be used in the same way as the appropriate standard library function, except they must be called from the appropriate namespace (for example, `cilk::__parallel` or `cilk::__parallel::__sort`). The standard library functions are intended to be run with an `ExecutionPolicy` parameter that determines whether the function will be executed in parallel. This version of the cilk implementation does not accept an `ExecutionPolicy` parameter and instead defaults to parallel execution. In general, these algorithms require iterators passed to them support random access.

# Style Notes
Most functions are named the same as the appropriate function in the C++ standard library. The exceptions would be differing implementations of the same function, such as `rotate` vs `rotate_inplace` or `find` vs `find2`. The intention is that the more performant version be used eventually, but for, multiple implementations are retained for completeness. Helper functions are demarcated as such in the function comment.
