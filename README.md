This script calculates the following Coulomb matrix elements using xsf file from wannier90:

```math
U = \int d\mathbf{r} d\mathbf{r'} \frac{W_1^2(\mathbf{r}) W_1^2(\mathbf{r'})}{|\mathbf{r} - \mathbf{r'}|} 
```

```math
V = \int d\mathbf{r} d\mathbf{r'} \frac{W_1^2(\mathbf{r}) W_2^2(\mathbf{r'})}{|\mathbf{r} - \mathbf{r'}|} 
```

```math
J = \int d\mathbf{r} d\mathbf{r'} \frac{W_1(\mathbf{r}) W_2(\mathbf{r}) W_1^2(\mathbf{r'}) W_2^2(\mathbf{r'}) }{|\mathbf{r} - \mathbf{r'}|} 
```
