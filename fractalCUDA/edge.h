/*
* Copyright 2018 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms 
* is strictly prohibited.
*
*/

#ifndef EDGE_H
#define EDGE_H

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////

//#define UMUL(a, b) ( (a) * (b) )
//#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

#define mymin(a, b) (((a) < (b)) ? (a) : (b)) 
#define mymax(a, b) (((a) > (b)) ? (a) : (b)) 
////////////////////////////////////////////////////////////////////////////////
// Reference CPU Kirsch edge detector
////////////////////////////////////////////////////////////////////////////////


extern "C" void kirschEdgeDetectorCPU(
	const unsigned char *data_in,
	const int *mask,
	unsigned char *data_out,
	const unsigned channels,
	const unsigned width,
	const unsigned height
);

////////////////////////////////////////////////////////////////////////////////
// GPU 
////////////////////////////////////////////////////////////////////////////////

extern "C" void kirschEdgeDetectorGPU(
	void *d_ImgDataIn,
	void *d_ImgMaskData,
	void *d_ImgDataOut,
	unsigned imgChannels,
	unsigned imgWidth,
	unsigned imgHeight
);


/* Kirsch matrices for convolution */
const int kirschFilter[8][3][3] = {
	{
		{ 5, 5, 5 },
		{ -3, 0, -3 },           /*rotation 1 */
		{ -3, -3, -3 }
	},
	{
		{ 5, 5, -3 },
		{ 5, 0, -3 },            /*rotation 2 */
		{ -3, -3, -3 }
	},
	{
		{ 5, -3, -3 },
		{ 5, 0, -3 },            /*rotation 3 */
		{ 5, -3, -3 }
	},
	{
		{ -3, -3, -3 },
		{ 5, 0, -3 },            /*rotation 4 */
		{ 5, 5, -3 }
	},
	{
		{ -3, -3, -3 },
		{ -3, 0, -3 },           /*rotation 5 */
		{ 5, 5, 5 }
	},
	{
		{ -3, -3, -3 },
		{ -3, 0, 5 },            /*rotation 6 */
		{ -3, 5, 5 }
	},
	{
		{ -3, -3, 5 },
		{ -3, 0, 5 },            /*rotation 7 */
		{ -3, -3, 5 }
	},
	{
		{ -3, 5, 5 },
		{ -3, 0, 5 },            /*rotation 8 */
		{ -3, -3, -3 }
	}
};


#endif
