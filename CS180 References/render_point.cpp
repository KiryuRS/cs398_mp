// render_point0.cpp
// jsh 1/08

#include "render_point.h"

COLORREF render_point(double x, double y, int N) {
  double a = 0.0, b = 0.0, norm2 = 0.0;
  int n;
  for (n = 0; norm2 < 4.0 && n < N; ++n) {
    double c = a*a - b*b + x;
    b = 2.0*a*b + y;
    a = c;
    norm2 = a*a + b*b;
  }
  int value = int(255*(1 - double(n)/N));
  return RGB(value,value,value);
}
