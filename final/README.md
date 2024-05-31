`examples.ipynb` is what you want to look at. 
`helpers.py` contains the loss function which outlines the steps I take during the forward pass.

This is an optimal control formulation of image registration.

$$
\min_v \frac{1}{2} \int_{\Omega} (\CT(z_x(T)) - \CR(x))^2 dx + S[v]
$$

subject to 

$$
\frac{d}{dt} z_x(t) = v(z_x(t),t), \quad \text{ for } \quad t in (0,T], \quad z_x(0)=x,
$$

In this discrete version, the continuous integral of the squared differences between the transformed template image and the reference image is approximated using the MSE as follows:

$$
\min_v \frac{1}{2N} \sum_{i=1}^N (\CT(z_{x_i}(T)) - \CR(x_i))^2 
$$

Here, $N$ represents the number of discrete grid points in the image domain $\Omega$. In the examples, I create $200 \times 200$ grid, so $40000$ grid points for the digit images which are quite small, $28 \times 28$. This approach normalizes the error by the number of points, making the image domain irrelavant and might result in slightly different behavior than original formulation. No regularization is used in my examples.


