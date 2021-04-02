#ifndef RNG_HPP
#define RNG_HPP

namespace RNG {
void init(unsigned long seed);
double uniform(double low = 0.0, double high = 1.0);
int uniform_int(int low, int high);
double normal(double mu = 0.0, double sigma = 1.0);
int poisson_int(double mu);

double uniform_unsplit(double low = 0.0, double high = 1.0);
int uniform_int_unsplit(int low, int high);
double normal_unsplit(double mu = 0.0, double sigma = 1.0);
int poisson_int_unsplit(double mu);
} // namespace RNG

#endif
