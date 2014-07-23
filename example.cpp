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
	printf("[%s][%4d]\tCost: %12.6f\tStep: %.4f\n", __TIME__, niter, cost, step);
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
	lbfgs_parameter_t *opt_param = (lbfgs_parameter_t *) vecmalloc(sizeof(lbfgs_parameter_t));
	lbfgs_parameter_init(opt_param);
	cudaDeviceSynchronize();
	opt_param->max_iterations = 200;

	int     dim     = 1;
	float   *ps     = (float *) vecalloc(sizeof(float));
	float   *theta  = (float *) vecalloc(dim * sizeof(float));
	float   cost;

	theta[0]    = -5.0f;    

	printf("Solve argmin [x^2 - 2x + 5] from initial point %f.\n", *theta);
	int ret = lbfgs(dim, theta, &cost, mycost, progress, (void*)(ps), opt_param);
	printf("L-BFGS optimization terminated with status code %d.\n", ret);
	cudaDeviceSynchronize();
	printf("The optimized point is %f.\n", *theta);
	system("pause");
	return 0;
}
