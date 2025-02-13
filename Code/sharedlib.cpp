#include <cmath>
#include <string>
#include <iostream>

extern "C"
{
    double y = M_PI;

    double compute_sin(double x)
    {
        return sin(x);
    }

    double compute_cos(double x) // New function
    {
        return cos(x);
    }

    double run()
    {
        return compute_sin(y);
    }
}
