#include "Common.h"

#include <complex>

// https://en.wikipedia.org/wiki/Newton_fractal

#define MAP(x, min, max) ( PIXELDIM * (x - min))/(max - min)
__forceinline float MapToMandrelbrot(float v, float min, float max)
{
  return v * (max - min) / (PIXELDIM - 1) + min;
}

__forceinline size_t Map(const double& x, const double& min, const double& max)
{
  return static_cast<size_t>(PIXELDIM * (x - min) / (max - min));
}

__forceinline void SetData(int x, int y, uchar* data, int color)
{
  //static constexpr double xMIN = -2.5f;
  //static constexpr double xMAX =  1.0f;
  //static constexpr double yMIN = -1.0f;
  //static constexpr double yMAX =  1.0f;
  //size_t index = Map(y, yMIN, yMAX) * PIXELDIM + Map(x, xMIN, xMAX);
  //size_t index = static_cast<size_t>((int)y * PIXELDIM + (int)x);
  size_t index = y * PIXELDIM + x;
  if (index < PIXELDIM2)
  {
    switch (color)
    {
    case 0:
      data[index + PIXELDIM2 + PIXELDIM2] = 0xff; // r
      data[index + PIXELDIM2] = 0x00;
      data[index] = 0x00;
      break;
    case 1:
      data[index + PIXELDIM2 + PIXELDIM2] = 0x00;
      data[index + PIXELDIM2] = 0xff; // g
      data[index] = 0x00;
      break;
    case 2:
      data[index + PIXELDIM2 + PIXELDIM2] = 0x00;
      data[index + PIXELDIM2] = 0x00;
      data[index] = 0xff; // b
      break;
    }
  }
}

std::complex<float> Fz(const std::complex<float>& z)
{
  return z*z*z - std::complex<float>(1.0f, 0.0f);
}

std::complex<float> dFz(const std::complex<float>& z)
{
  return std::complex<float>(3.0f, 0.0f) * (z*z);
}

//float Fz(const float& z)
//{
//  return z * z * z - float(1.0f);
//}
//
//float dFz(const float& z)
//{
//  return 3.0f * (z * z);
//}

struct Vec2
{
  float x;
  float y;

  Vec2 operator-(const Vec2& rhs)
  {
    Vec2 tmp;
    tmp.x -= rhs.x;
    tmp.y -= rhs.y;
    return *this;
  }
};

void NewtonCPU(uchar * data)
{
  for (auto y = 0; y < PIXELDIM; ++y)
  {
    float zy = ceilf((float)y * 2.0f / (PIXELDIM) + -1.0f);
    for (auto x = 0; x < PIXELDIM; ++x)
    {
      float zx = ceilf((float)x * 2.0f / (PIXELDIM) + -1.0f);

      // Mapped coordinates
      std::complex<float> z{ zx, zy };

      // Roots of polynomials
      std::complex<float> roots[3] =
      {
        std::complex<float>{ 1.0f, 0.0f },
        std::complex<float>{ -0.5f,  sqrtf(3.0f) / 2.0f},
        std::complex<float>{ -0.5f, -sqrtf(3.0f) / 2.0f},
      };

      int iteration     = 0;

      bool done = false;
      while (iteration < MAX_ITERATIONS)
      {
        // Newton-Raphson
        z -= Fz(z) / dFz(z);

        for (int i = 0; i < 3; ++i)
        {
          std::complex<float> diff = (z - roots[i]);

          if (std::fabsf(diff.real()) < EPSILON && std::fabsf(diff.imag()) < EPSILON)
          {
            SetData(x, y, data, i);
            done = true;
            continue;
          }
        }

        ++iteration;
      }
      if (!done)
        SetData(x, y, data, 3);
    }
  }

  //for (auto x = 0; x < PIXELDIM; ++x)
  //{
  //  for (auto y = 0; y < PIXELDIM; ++y)
  //  {
  //    float zx = MapToMandrelbrot((float)x, -2.5f, 1.0f);
  //    float zy = MapToMandrelbrot((float)y, -1.0f, 1.0f);

  //    // Mapped coordinates
  //    Vec2 z{ zx, zy };

  //    // Roots of polynomials
  //    Vec2 roots[3] =
  //    {
  //      Vec2{ 1.0f, 0.0f },
  //      Vec2{ -0.5f,  sqrtf(3.0f) / 2.0f},
  //      Vec2{ -0.5f, -sqrtf(3.0f) / 2.0f},
  //    };

  //    int iteration = 0;
  //    constexpr int maxIteration = 1;

  //    bool done = false;
  //    while (iteration < maxIteration)
  //    {
  //      // Newton-Raphson
  //      z -= Fz(z) / dFz(z);

  //      for (int i = 0; i < 3; ++i)
  //      {
  //        Vec2 diff = (z - roots[i]);

  //        if (std::fabsf(diff.real()) < EPSILON && std::fabsf(diff.imag()) < EPSILON)
  //        {
  //          SetData(x, y, data, i);
  //          done = true;
  //          continue;
  //        }
  //      }

  //      ++iteration;
  //    }
  //    if (!done)
  //      SetData(x, y, data, 3);
  //  }
  //}
};
