#include "SierpinskiTriangle.h"
#include <iostream>


//N must be in 2 power x
#define N 1<<15




using namespace std;
void TriangleCPU(uchar* data)
{
	std::ofstream outFile{ "result.txt" };
	int value = 0;
	if (outFile.is_open())
	{
		 for (int y = N - 1; y >= 0; y--)
		 {

		 	// printing space till 
		 	// the value of y 
		 	for (int i = 0; i < y; i++)
		 	{
		 		outFile << " ";
		 
		 	}
		 	// printing '*' 
		 	for (int x = 0; x + y < N; x++)
		 	{

		 		// printing '*' at the appropriate position 
		 		// is done by the and value of x and y 
		 		// wherever value is 0 we have printed '*' 
		 		if (x & y)
		 		{
		 			 outFile << " " << " ";
		 		
		 		}
		 		else
		 		{
		 			outFile << "* ";
		 		
		 		}

		 	}

		 	outFile << endl;
	
		 }
		 outFile.close();

	}
				

	
}

void TriangleGPU(uchar* data)
{

}


