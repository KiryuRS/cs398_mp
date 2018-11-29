#include "BrownianTree.h"

void BrownianSetData(uint x, uint y, uchar value, uchar* data)
{
	data[x + PIXELDIM * y] = value;								// b
	data[x + PIXELDIM * y + PIXELDIM2] = value;					// g
	data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value;		// r
}

void BrownianCPU(uchar* data)
{
	uchar *input = new uchar[PIXELDIM * PIXELDIM]{ };
	srand((unsigned)time(nullptr));
	int px, py; // particle values
	int dx, dy; // offsets

	// set the seed
	input[(rand() % PIXELDIM) * PIXELDIM + rand() % PIXELDIM] = 1;

	for (int i = 0; i != 100000; ++i) {
		// set particle's initial position
		px = rand() % PIXELDIM;
		py = rand() % PIXELDIM;

		while (1) {
			// randomly choose a direction
			dx = rand() % 3 - 1;
			dy = rand() % 3 - 1;

			if (dx + px < 0 || dx + px >= PIXELDIM || dy + py < 0 || dy + py >= PIXELDIM) {
				// plop the particle into some other random location
				px = rand() % PIXELDIM;
				py = rand() % PIXELDIM;
			}
			else if (input[(py + dy) * PIXELDIM + px + dx] != 0) {
				// bumped into something
				input[py * PIXELDIM + px] = 1;
				break;
			}
			else {
				py += dy;
				px += dx;
			}
		}
	}

	for (uint i = 0; i != PIXELDIM; ++i)
		for (uint j = 0; j != PIXELDIM; ++j)
		{
			uchar rgb = input[i * PIXELDIM + j] ? 255 : 0;
			BrownianSetData(i, j, rgb, data);
		}

	delete[] input;
}
