This script calculates the bare Coulomb matrix elements using real space Wannier functions (xsf files) $W_i(\mathbf{r})$ from wannier90 via Monte Carlo sampling:

```math
U = \int d\mathbf{r} d\mathbf{r'} \frac{W_1^2(\mathbf{r}) W_1^2(\mathbf{r'})}{|\mathbf{r} - \mathbf{r'}|} 
```

```math
V = \int d\mathbf{r} d\mathbf{r'} \frac{W_1^2(\mathbf{r}) W_2^2(\mathbf{r'})}{|\mathbf{r} - \mathbf{r'}|} 
```

```math
J = \int d\mathbf{r} d\mathbf{r'} \frac{W_1(\mathbf{r}) W_2(\mathbf{r}) W_1(\mathbf{r'}) W_2(\mathbf{r'}) }{|\mathbf{r} - \mathbf{r'}|} 
```

# Dependencies
**python** version requires [numba](https://numba.pydata.org) \
**c++** version can be complied with OpenMP via CMake:\
mkdir build \
cd build/ \
cmake .. \
make 


# Usage
Run python Wannier_Coulomb.py or Wannier_Coulomb.x in the same directory with W1.xsf and W2.xsf files from [wannier90](https://wannier.org).
Set r_center and r_cut to reduce the size for Monte Carlo sampling, but keeping the norms of Wannier functions close to 1. 

