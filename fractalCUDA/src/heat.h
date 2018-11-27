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

#ifndef HEAT_H
#define HEAT_H

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// Reference CPU version  
////////////////////////////////////////////////////////////////////////////////
extern "C" void initPoints(
	float *pointIn,
	float *pointOut,
	uint nRowPoints
);

extern "C" void heatDistrCPU(
	float *pointIn,
	float *pointOut,
	uint nRowPoints,
	uint nIter
);

////////////////////////////////////////////////////////////////////////////////
// GPU version 
////////////////////////////////////////////////////////////////////////////////

extern "C" void heatDistrGPU(
	float *d_DataIn,
	float *d_DataOut,
	uint nRowPoints,
	uint nIter
);

#endif
