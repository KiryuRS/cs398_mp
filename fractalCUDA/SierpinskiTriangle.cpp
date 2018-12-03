#include "SierpinskiTriangle.h"
#include <iostream>


//N must be in 2 power x 
#define N (1<<9)
static constexpr size_t  N2 = (N*N);
__forceinline size_t Map(const double& x, const double& min, const double& max)
{
	return static_cast<size_t>(PIXELDIM * (x - min) / (max - min));
}
void SetData( int x, int y,int value, uchar* data)
{
	

	if (value == 0)
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value; // r
	}
	
	else
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value; // r
	}

}


using namespace std;
void TriangleCPU(uchar* data)
{
	
	for (int y = N - 1; y >= 0; y--)
	{
	
		// printing space till 
		// the value of y 
		for (int i = 0; i < y; i++)
		{
			//outFile << " ";
			SetData(i, y,255 ,data);
	
		}
		// printing '*' 
		for (int x = 0; x + y < N; x++)
		{
	
			// printing '*' at the appropriate position 
			// is done by the and value of x and y 
			// wherever value is 0 we have printed '*' 
			if (x & y)
			{
				//outFile << " " << " ";
				SetData(x, y,  255, data);
				//SetData(x, y, 255, data);
			}
			else
			{
				SetData(x, y,0, data);
				//outFile << "* "; 
	
			}
	
		}
		
		//outFile << endl; 
	}
				
	
	
}

void TriangleGPU(uchar* CPUin, uchar* GPUout)
{

}


