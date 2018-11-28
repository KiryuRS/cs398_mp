// mandelbrot.cpp
// -- Mandelbrot set demo
// cs180 1/08

#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <windows.h>
#include "render_point.h"


class Mandelbrot {
  public:
    Mandelbrot(HINSTANCE, int);
    void ComputeNewWindowRect(RECT&, HWND win, const RECT&);
    void Render(HWND);
    void Draw(HDC, const RECT&);
    ~Mandelbrot(void);
    static int Clip(int x, int min, int max);
    static void Normalize(RECT &);
  private:
    const char *name;
    static LRESULT CALLBACK WinProc(HWND, UINT, WPARAM, LPARAM);
    HINSTANCE hinstance;
    HBITMAP back_buffer;
    HDC buffer_dc;
    double window_area;
    double lower_left_x, lower_left_y, scale;
};


int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR, int show)
{
  Mandelbrot *mandelbrot = new Mandelbrot(hinst,show);

  MSG msg;
  while (GetMessage(&msg,0,0,0))
  {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  
  delete mandelbrot;
  return msg.wParam;
}


Mandelbrot::Mandelbrot(HINSTANCE hinst, int show)
    : hinstance(hinst), name("Mandelbrot Set Demo"),
      back_buffer(0) {

  WNDCLASS wc;
  wc.style = 0;
  wc.lpfnWndProc = WinProc;
  wc.cbClsExtra = 0;
  wc.cbWndExtra = sizeof(Mandelbrot*);
  wc.hInstance = hinstance;
  wc.hIcon = LoadIcon(0,IDI_APPLICATION);
  wc.hCursor = LoadCursor(0,IDC_ARROW);
  wc.hbrBackground = 0;
  wc.lpszMenuName = 0;
  wc.lpszClassName = name;
  RegisterClass(&wc);
  HWND win = CreateWindow(name,name,WS_CAPTION|WS_SYSMENU,
                          CW_USEDEFAULT,CW_USEDEFAULT,
                          CW_USEDEFAULT,CW_USEDEFAULT,
                          0,0,hinstance,this);

  window_area = 0.25f*GetSystemMetrics(SM_CXSCREEN)
                    *GetSystemMetrics(SM_CYSCREEN);
  int length = int(std::sqrt(window_area));
  RECT rect;
  rect.left = (GetSystemMetrics(SM_CXSCREEN) - length)/2;
  rect.top = (GetSystemMetrics(SM_CYSCREEN) - length)/2;
  rect.right = rect.left + length;
  rect.bottom = rect.top + length;
  lower_left_x = -1.5f;
  lower_left_y = -1.0f;
  scale = 2.0f/length;
    AdjustWindowRect(&rect,WS_CAPTION|WS_SYSMENU,FALSE);
  MoveWindow(win,rect.left,rect.top,rect.right-rect.left,
             rect.bottom-rect.top,FALSE);
  ShowWindow(win,show);
  Render(win);
}


Mandelbrot::~Mandelbrot(void) {
  if (back_buffer) {
      DeleteDC(buffer_dc);
      DeleteObject(back_buffer);
  }
  UnregisterClass(name,hinstance);
}


LRESULT CALLBACK Mandelbrot::WinProc(HWND win, UINT msg, WPARAM wp, LPARAM lp) {
  static bool mouse_moving = false;
  static RECT rect_select;
  static Mandelbrot *mp = 0;

  switch (msg) {

    case WM_CREATE: {
      CREATESTRUCT *csp = reinterpret_cast<CREATESTRUCT*>(lp);
      mp = reinterpret_cast<Mandelbrot*>(csp->lpCreateParams);
      return 0; }

    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC dc = BeginPaint(win,&ps);
      mp->Draw(dc,ps.rcPaint);
      EndPaint(win,&ps);
      return 0; }

    case WM_LBUTTONDOWN:
        mouse_moving = true;
        rect_select.left = rect_select.right = LOWORD(lp);
        rect_select.top = rect_select.bottom = HIWORD(lp);
        SetCapture(win);
        return 0;

    case WM_MOUSEMOVE:
        if (mouse_moving) {
          HDC dc = GetDC(win);
          SetROP2(dc,R2_XORPEN);
          Rectangle(dc,rect_select.left,rect_select.top,
                       rect_select.right,rect_select.bottom);
          RECT rect;
          GetClientRect(win,&rect);
          rect_select.right = Clip(short(LOWORD(lp)),rect.left,rect.right);
          rect_select.bottom = Clip(short(HIWORD(lp)),rect.top,rect.bottom);
          Rectangle(dc,rect_select.left,rect_select.top,
                       rect_select.right,rect_select.bottom);
          ReleaseDC(win,dc);
      }
        return 0;

    case WM_LBUTTONUP:
      if (mouse_moving) {
          mouse_moving = false;
          ReleaseCapture();
        Normalize(rect_select);
          int dy = rect_select.bottom - rect_select.top,
              dx = rect_select.right - rect_select.left;
          if (dx > 4 && dy > 4) {
          RECT rect;
          mp->ComputeNewWindowRect(rect,win,rect_select);
          MoveWindow(win,rect.left,rect.top,rect.right-rect.left,
                     rect.bottom-rect.top,TRUE);
          mp->Render(win);
        }
      }
        return 0;

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;

  }
  return DefWindowProc(win,msg,wp,lp);
}


void Mandelbrot::ComputeNewWindowRect(RECT &rect, HWND win,
                                      const RECT& sub_rect) {
  double ratio = double(sub_rect.bottom-sub_rect.top)
                /double(sub_rect.right-sub_rect.left);
  GetClientRect(win,&rect);
  lower_left_x += scale*(sub_rect.left-rect.left);
  lower_left_y += scale*(rect.bottom-sub_rect.bottom);
    POINT corner = {0,0};
    ClientToScreen(win,&corner);
  rect.left = corner.x;
  rect.top = corner.y;
  rect.right =  corner.x + int(std::sqrt(window_area/ratio));
    rect.bottom = corner.y + int(std::sqrt(window_area*ratio));
  scale = scale*double(sub_rect.right-sub_rect.left)
                /double(rect.right-rect.left);
    AdjustWindowRect(&rect,WS_CAPTION|WS_SYSMENU,FALSE);
}


int Mandelbrot::Clip(int x, int min, int max) {
  int y = (x < min) ? min : x;
  return (y > max) ? max : y;
}


void Mandelbrot::Normalize(RECT &rect) {
  if (rect.left > rect.right) {
    int temp = rect.left;
    rect.left = rect.right;
    rect.right = temp;
  }
  if (rect.top > rect.bottom) {
    int temp = rect.top;
    rect.top = rect.bottom;
    rect.bottom = temp;
  }
}


void Mandelbrot::Draw(HDC dc, const RECT& rect) {
  if (back_buffer) {
      BitBlt(dc,rect.left,rect.top,rect.right-rect.left,
             rect.bottom-rect.top,
             buffer_dc,rect.left,rect.top,SRCCOPY);
  }
}


void Mandelbrot::Render(HWND win) {
  if (back_buffer) {
      DeleteDC(buffer_dc);
      DeleteObject(back_buffer);
  }
  RECT rect;
  GetClientRect(win,&rect);
  HDC dc = GetDC(win);
  buffer_dc = CreateCompatibleDC(dc);
  back_buffer = CreateCompatibleBitmap(dc,rect.right,rect.bottom);
  SelectObject(buffer_dc,back_buffer);
  SetWindowText(win,name);
  double time = std::clock();
  for (int j=0; j < rect.right; ++j) {
    for (int i=0; i < rect.bottom; ++i) {
      double x = lower_left_x + j*scale,
            y = lower_left_y + scale*(rect.bottom-i);
      COLORREF color = render_point(x,y,5120);
      SetPixel(buffer_dc,j,i,color);
    }
    BitBlt(dc,j,0,1,rect.bottom,buffer_dc,j,0,SRCCOPY);
  }
  time = (std::clock() - time)/double(CLOCKS_PER_SEC);
  std::stringstream str;
  str << std::fixed << std::setprecision(2);
  str << name << ": " << time << " seconds";
  SetWindowText(win,str.str().c_str());
  ReleaseDC(win,dc);
}
