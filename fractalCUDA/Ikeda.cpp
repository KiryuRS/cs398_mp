#include "Ikeda.h"

// https://en.wikipedia.org/wiki/Ikeda_map

static constexpr float u = 0.918f;

inline float x_next(float x, float y, float t)
{
  return 1 + u * (x * std::cosf(t) - y * std::sinf(t));
}

inline float y_next(float x, float y, float t)
{
  return u * (x * std::sinf(t) + y * std::cosf(t));
}

inline float t_next(float x, float y)
{
  return 0.4f - 6.0f / (1 + (x * x) + (y * y));
}

__forceinline void SetIkeda(int x, int y, uchar* data)
{
  size_t index = y * PIXELDIM + x;
  if (index < PIXELDIM2)
  {
    data[index + PIXELDIM2 + PIXELDIM2] = 0x00;
    data[index + PIXELDIM2] = 0x00;
    data[index] = 0x00;
  }
}

void IkedaCPU(uchar * data)
{
  float min_x = -0.5f, min_y = -2.5f;
  float max_x = 6.5f, max_y = 6.5f;

  //float min_x = FLT_MAX, min_y = FLT_MAX;
  //float max_x = FLT_MIN, max_y = FLT_MIN;
  for (auto y = 0; y < PIXELDIM; y += 1)
  {
    for (auto x = 0; x < PIXELDIM; x += 1)
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

        //dataSet[(uint)ceilf(zy) * PIXELDIM + (uint)ceilf(zx)] = 1;
        //std::cout << zx << "   " << zy << std::endl;

        if (iteration % 10 == 0 && iteration > 100)
        {
          //min_x = zx < min_x ? zx : min_x;
          //min_y = zy < min_y ? zy : min_y;
          //max_x = zx > max_x ? zx : max_x;
          //max_y = zy > max_y ? zy : max_y;
          SetIkeda((int)ceilf((PIXELDIM * (zx - min_x)) / (max_x - min_x)), (int)ceilf((PIXELDIM * (zy - min_y)) / (max_y - min_y)), data);
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
