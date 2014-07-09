#include "lbfgs.h"
#include "arith.h"

// printing the optimization process
static int progress(void        *instance,
                    const real  *theta,
                    const real  *grad,
                    const real  cost,
                    const real  normTheta,
                    const real  normGrad,
                    const real  step,
                    int         nparam,
                    int         niter,
                    int         ls)
{
    printf("[%4d]\tCost: %12.6f\tStep: %.4f\n", niter, cost, step);
    return 0;
}

// solve the problem: argmin f(x) = x^2 - 2*x + 5
// function returns cost, optimized x saved in theta
// use your own cost function instead
float costfn(void           *config,        // configuration
             const float    *theta,         // x, to be optimized
             float          *grad,          // gradients
             const int      n, 
             const float    step)
{
    float cost, x;
    x = *theta;
    cost = x * x - 2 *x + 5;
    *grad = 2 * x - 2;

    return cost;
}

int main()
{
    lbfgs_parameter_t *opt_param;
    cudaMallocManaged((void **) &opt_param, sizeof(lbfgs_parameter_t));
    lbfgs_parameter_init(opt_param);
    cudaDeviceSynchronize();
    opt_param->max_iterations = 200;

    int     dim     = 1;
    float   *theta  = (float *) vecmalloc(dim * sizeof(float));
    float   *ps     = (float *) vecmalloc(sizeof(float));
    float   cost    = 0.f;    

    theta[0] = 3.0f;

    int ret = lbfgs(dim, theta, &cost, costfn, progress, (void*)(ps), opt_param);
    printf("L-BFGS optimization terminated with status code %d.\n", ret);
    return 0;
}
