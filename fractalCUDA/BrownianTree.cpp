#include "BrownianTree.h"

void DrawBrownianTree(uchar* data) {
	int px, py; // particle values
	int dx, dy; // offsets
	int i;

	// set the seed
	data[(rand() % PIXELDIM) * PIXELDIM + rand() % PIXELDIM] = 1;

	for (i = 0; i < NUM_PARTICLES; i++) {
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
			else if (data[(py + dy) * PIXELDIM + px + dx] != 0) {
				// bumped into something
				data[py * PIXELDIM + px] = 1;
				break;
			}
			else {
				py += dy;
				px += dx;
			}
		}
	}
}

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

void BrownianCPU(uchar* data)
{
	uchar* cpuIn = new uchar[PIXELDIM * PIXELDIM];
	srand((unsigned)time(nullptr));

	DrawBrownianTree(cpuIn);

	for (uint y = 0; y != PIXELDIM; ++y)
		for (uint x = 0; x != PIXELDIM; ++x)
		{

		}
}
