#include "henon.h"

#include <iostream>

struct bits24
{
    char s[24] = { 0 };
};

//#define CLAMP(x) x < 0.0 ? 0.0 : x > (double)PIXELDIM2 ? (double)PIXELDIM2 : x
#define CLAMP(x) x
#define MAP(x, min, max) ( PIXELDIM * (x - min))/(max - min)

__forceinline size_t Map(const double& x, const double& min, const double& max)
{
  return static_cast<size_t>(PIXELDIM * (x - min) / (max - min));
}

__forceinline void SetData(const double& x, const double& y, uchar* data)
{
    static constexpr double xMIN = -1.41;
    static constexpr double xMAX = 1.41;
    static constexpr double yMIN = -0.42;
    static constexpr double yMAX = 0.42;
    size_t index = Map(y, yMIN, yMAX) * PIXELDIM + Map(x, xMIN, xMAX);
    if (index < PIXELDIM2)
    {
      data[index] = 0x00; // b
      data[index + PIXELDIM2] = 0x00; // g
      data[index + PIXELDIM2 + PIXELDIM2] = 0xff; // r
    }
}

void HenonCPU(uchar * data)
{
    static constexpr size_t iteration = 10000;
    static constexpr double alpha = 1.4;
    static constexpr double beta  = 0.3;
    double x = 0.1;
    double y = 0.3;
    //*reinterpret_cast<bits24*>(data[GetIndex(x, y)]) = bits24{};
    SetData(x, y, data);

    //double xmin = 1.0;
    //double xmax = 1.0;
    //double ymin = 1.0;
    //double ymax = 1.0;

    //for (auto i = 0u; i < PIXELDIM3 / 3; ++i)
    //{
    //    data[i + 0] = UCHAR_MAX;
    //    data[i + 1] = UCHAR_MAX;
    //    data[i + 2] = UCHAR_MAX;
    //}
    for (auto i = 0u; i < iteration; ++i)
    {
        double xN = 1.0 - alpha * x * x + y;
        double yN = beta * x;
        SetData(xN, yN, data);
        x = xN, y = yN;

        // min = xN < yN ? (xN < min ? xN : min) : (yN < min ? yN : min);
        // max = xN > yN ? (xN > max ? xN : max) : (yN > max ? yN : max);
        //xmin = xN < xmin ? xN : xmin;
        //xmin = xN < xmin ? xN : xmin;
        //ymin = yN < ymin ? yN : ymin;
        //ymin = yN < ymin ? yN : ymin;
    }
}
