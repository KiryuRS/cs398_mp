#include "SierpinskiTriangle.h"
#include <iostream>


//N must be in 2 power x 
#define N (1<<9)



void STriangle::TriangleCPU(uchar* data)
{
	
	for (int y = N - 1; y >= 0; y--)
	{
	
		// printing space till 
		// the value of y 
		for (int i = 0; i < y; i++)
		{
			//outFile << " ";
			//SetData(i, y,255 ,data);
			data[i + PIXELDIM * y] = 0XFF; // b
			data[i + PIXELDIM * y + PIXELDIM2] = 0XFF; // g
			data[i + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0XFF; // r
	
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
				//SetData(x, y,  255, data);
				data[x + PIXELDIM * y] = 0XFF; // b
				data[x + PIXELDIM * y + PIXELDIM2] = 0XFF; // g
				data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0XFF; // r
				//SetData(x, y, 255, data);
			}
			else
			{
				//SetData(x, y,0, data);
				data[x + PIXELDIM * y] = 0x00; // b
				data[x + PIXELDIM * y + PIXELDIM2] = 0x00; // g
				data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0XFF; // r
				//outFile << "* "; 

			}
	
		}
		
		//outFile << endl; 
	}
				
	
	
}



