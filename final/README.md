This is an optimal control formulation of image registration.

$$
\min_v \frac{1}{2} \int_{\Omega} (\CT(z_x(T)) - \CR(x))^2 dx + S[v]
$$

subject to 

$$
\frac{d}{dt} z_x(t) = v(z_x(t),t), \quad \text{ for } \quad t \in (0,T], \quad z_x(0)=x,
$$


The integral is approximated by evaluating the transformed template $\CT(z_x(T))$ and the reference $\CR(x)$ at grid points that can be varied in width. In the reading, the integral can instead be approximated with a quadrature rule, so that is what I will experiment with next (look at [torchquad](https://github.com/esa/torchquad)).