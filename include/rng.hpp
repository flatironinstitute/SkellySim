#ifndef RNG_HPP
#define RNG_HPP

namespace RNG {
void init(unsigned long seed);
double uniform(double low = 0.0, double high = 1.0);
double normal(double mu = 0.0, double sigma = 1.0);
double uniform_unsplit(double low = 0.0, double high = 1.0);
double normal_unsplit(double mu = 0.0, double sigma = 1.0);
} // namespace RNG

#endif
