/******************************************************************************/
/*!
@file   SierpinskiCarpet.h
@par    Purpose: Header file for SierpinskiCarpet
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Kenneth
@par    Email: t.weigangkenneth\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once
#include "Common.h"
#include <cstdio>

#define SIERPINSKI_DEPTH		4
#define WHITESPACE_PRINT		1 << 1
#define HEX_PRINT				1 << 2
#define NEWLINE_PRINT			1 << 3
//#define SC_UNIFIED			100
#define SC_PINNED				101
//#define SC_MANAGED			102

void SierpinskiCarpetCPU(uchar *data);
void SierpinskiCarpetGPU(uchar *cpuData, uchar **gpuData);
void SierpinskiCarpetKernel(uchar *d_DataIn, uint width, uint height);
void SierpinskiCarpetFree();

class FileWriter
{
	FILE* f;

public:
	FileWriter(const std::string& filename)
		: f{ nullptr }
	{
		f = fopen(filename.c_str(), "w");
		if (!f)
			std::cerr << "Something went wrong!" << std::endl;
	}

	FileWriter(const FileWriter&) = delete;
	FileWriter(FileWriter&&) = delete;
	FileWriter& operator=(const FileWriter&) = delete;
	FileWriter& operator=(FileWriter&&) = delete;

	void Write(const std::string& str)
	{
		if (f)
			fprintf(f, str.c_str());
	}

	~FileWriter()
	{
		if (f)
			fclose(f);
	}
};