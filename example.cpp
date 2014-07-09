#include "lbfgs.h"
#include "arith.h"

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
float mycost(void *config, const float *theta, float *grad, const int n, const float step)
{
    float cost, x, y;
    x = theta[0];
    cost = x * x - 2 *x + 5;
    grad[0] = 2 * x - 2;
    return cost;
}
int main()
{
    lbfgs_parameter_t *opt_param;
    cudaMallocManaged((void **) &opt_param, sizeof(lbfgs_parameter_t));
    lbfgs_parameter_init(opt_param);
    cudaDeviceSynchronize();
    opt_param->max_iterations = 200;

    int dim = 1;
    float *theta = (float *) vecmalloc(dim * sizeof(float));
    theta[0] = 3.0f;
    float cost = 0.f;
    float *ps = (float *) vecmalloc(sizeof(float));

    int ret = lbfgs(dim, theta, &cost, mycost, progress, (void*)(ps), opt_param);
    printf("L-BFGS optimization terminated with status code %d.\n", ret);
    return 0;
}