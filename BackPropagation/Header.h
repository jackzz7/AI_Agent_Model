#pragma once
#include<assert.h>

struct Pixel {
	//0-1==>0,255
	float r, g, b;
	Pixel() :r(0), g(0), b(0) {}
	Pixel(double r, double g, double b) :r(r), g(g), b(b) {
		assert(abs(r) <= 1 && abs(g) <= 1 && abs(b) <= 1);
	}
};