#include "henon.h"

#include <iostream>

struct bits24
{
    char s[24] = { 0 };
};

__forceinline size_t GetIndex(const double& x, const double& y)
{
    return static_cast<size_t>(y * PIXELDIM + x) % (PIXELDIM3 - 2);
}

void HenonCPU(uchar * data)
{
    static constexpr size_t iteration = 10000;
    static constexpr double alpha = 1.4;
    static constexpr double beta  = 0.3;
    double x = 1.0f;
    double y = 1.0f;
    //*reinterpret_cast<bits24*>(data[GetIndex(x, y)]) = bits24{};
    data[10 + 0] = UCHAR_MAX;
    data[10 + 1] = UCHAR_MAX;
    data[10 + 2] = 0x0;

    for (auto i = 0u; i < PIXELDIM3 / 3; ++i)
    {
        data[i + 0] = UCHAR_MAX;
        data[i + 1] = UCHAR_MAX;
        data[i + 2] = UCHAR_MAX;
    }
    //for (auto i = 0u; i < iteration; ++i)
    //{
    //    double xN = 1.0 - alpha * (x * x) + y;
    //    double yN = beta * x;
    //    data[GetIndex(x, y) + 0] = 0xff;
    //    data[GetIndex(x, y) + 1] = 0xff;
    //    data[GetIndex(x, y) + 2] = 0xff;
    //    x = xN, y = yN;
    //}
}
