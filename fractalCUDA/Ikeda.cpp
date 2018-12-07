/******************************************************************************/
/*!
@file   Ikeda.cpp
@par    Purpose: Implementation of Ikeda
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Alvin
@par    Email: alvin.tan\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "Ikeda.h"

// https://en.wikipedia.org/wiki/Ikeda_map

#if defined (DOUBLE_VERSION)
static constexpr double u = 0.918;

inline double x_next(double x, double y, double t)
{
  return 1.0 + u * (x * std::cos(t) - y * std::sin(t));
}

inline double y_next(double x, double y, double t)
{
  return u * (x * std::sin(t) + y * std::cos(t));
}

inline double t_next(double x, double y)
{
  return 0.4 - 6.0 / (1.0 + (x * x) + (y * y));
}

__forceinline void SetIkeda(int x, int y, uchar* data)
{
  size_t index = y * PIXELDIM + x;
  if (index < PIXELDIM2)
  {
    data[index + PIXELDIM2 + PIXELDIM2] = 0xff;
    data[index + PIXELDIM2] = 0x00;
    data[index] = 0x00;
  }
}

void IkedaCPU(uchar * data)
{
  static constexpr double min_x = -0.5, min_y = -2.5;
  static constexpr double max_x = 6.5, max_y = 6.5;

  //double min_x = FLT_MAX, min_y = FLT_MAX;
  //double max_x = FLT_MIN, max_y = FLT_MIN;
  for (auto y = 0; y < PIXELDIM; ++y)
  {
    for (auto x = 0; x < PIXELDIM; ++x)
    {
      double zx = (double)x;
      double zy = (double)y;

      int iteration = 0;
      while (iteration < MAX_ITERATIONS_IKEDA)
      {
        double t = t_next(zx, zy);
        double x_n = x_next(zx, zy, t);
        double y_n = y_next(zx, zy, t);
        zx = x_n;
        zy = y_n;

        //dataSet[(uint)ceilf(zy) * PIXELDIM + (uint)ceilf(zx)] = 1;
        //std::cout << zx << "   " << zy << std::endl;

        if (iteration > 100)
        {
          //min_x = zx < min_x ? zx : min_x;
          //min_y = zy < min_y ? zy : min_y;
          //max_x = zx > max_x ? zx : max_x;
          //max_y = zy > max_y ? zy : max_y;
          SetIkeda((int)ceil((PIXELDIM * (zx - min_x)) / (max_x - min_x)), (int)ceil((PIXELDIM * (zy - min_y)) / (max_y - min_y)), data);
        }

        ++iteration;
      }

      //zx = ceilf((PIXELDIM * (zx - min_x)) / (max_x - min_x));
      //zy = ceilf((PIXELDIM * (zy - min_y)) / (max_y - min_y));

    }
  }
  //std::cout << min_x << '\n'
  //          << max_x << '\n'
  //          << min_y << '\n'
  //          << max_y << std::endl;
}
#elif defined (FLOAT_VERSION)
static constexpr float u = 0.918f;

inline float x_next(float x, float y, float t)
{
	return 1.0f + u * (x * std::cosf(t) - y * std::sinf(t));
}

inline float y_next(float x, float y, float t)
{
	return u * (x * std::sinf(t) + y * std::cosf(t));
}

inline float t_next(float x, float y)
{
	return 0.4f - 6.0f / (1.0f + (x * x) + (y * y));
}

__forceinline void SetIkeda(int x, int y, uchar* data)
{
	size_t index = y * PIXELDIM + x;
	if (index < PIXELDIM2)
	{
		data[index + PIXELDIM2 + PIXELDIM2] = 0xff;
		data[index + PIXELDIM2] = 0x00;
		data[index] = 0x00;
	}
}

void IkedaCPU(uchar * data)
{
	static constexpr float min_x = -0.5f, min_y = -2.5f;
	static constexpr float max_x = 6.5f, max_y = 6.5f;

	for (auto y = 0; y < PIXELDIM; ++y)
	{
		for (auto x = 0; x < PIXELDIM; ++x)
		{
			float zx = (float)x;
			float zy = (float)y;

			int iteration = 0;
			while (iteration < MAX_ITERATIONS_IKEDA)
			{
				float t = t_next(zx, zy);
				float x_n = x_next(zx, zy, t);
				float y_n = y_next(zx, zy, t);
				zx = x_n;
				zy = y_n;

				if (iteration > 100)
					SetIkeda((int)ceilf((PIXELDIM * (zx - min_x)) / (max_x - min_x)), (int)ceilf((PIXELDIM * (zy - min_y)) / (max_y - min_y)), data);

				++iteration;
			}
		}
	}
}
#endif