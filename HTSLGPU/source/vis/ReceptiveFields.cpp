/*
HTSLGPU
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "ReceptiveFields.h"

using namespace vis;

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

void ReceptiveFields::create(const htsl::RecurrentSparseCoder2D &rsc2d) {
	int rootRfSize = rsc2d.getReceptiveRadius() * 2 + 1;

	_image.create(rsc2d.getWidth() * rootRfSize, rsc2d.getHeight() * rootRfSize);
}

void ReceptiveFields::render(const htsl::RecurrentSparseCoder2D &rsc2d, sys::ComputeSystem &cs) {
	int rootRfSize = rsc2d.getReceptiveRadius() * 2 + 1;
	int rfSize = rootRfSize * rootRfSize;
	
	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> weightDims;
	weightDims[0] = rsc2d.getWidth();
	weightDims[1] = rsc2d.getHeight();
	weightDims[2] = rfSize;

	std::vector<float> weightData(weightDims[0] * weightDims[1] * weightDims[2]);

	cs.getQueue().enqueueReadImage(rsc2d._hiddenVisibleWeights, CL_TRUE, zeroCoord, weightDims, 0, 0, weightData.data());

	float minimum = 99999.0f;
	float maximum = -99999.0f;

	for (int rx = 0; rx < rsc2d.getWidth(); rx++)
		for (int ry = 0; ry < rsc2d.getHeight(); ry++) {
			for (int fx = 0; fx < rootRfSize; fx++)
				for (int fy = 0; fy < rootRfSize; fy++) {
					int index = (rx + ry * rsc2d.getWidth()) + (fy + fx * rootRfSize) * rsc2d.getWidth() * rsc2d.getHeight();

					minimum = std::min(minimum, weightData[index]);
					//minimum = std::min(minimum, weightData[index].y);
					maximum = std::max(maximum, weightData[index]);
					//maximum = std::max(maximum, weightData[index].y);
				}
		}

	float mult = 255.0f / (maximum - minimum);

	for (int rx = 0; rx < rsc2d.getWidth(); rx++)
		for (int ry = 0; ry < rsc2d.getHeight(); ry++) {
			for (int fx = 0; fx < rootRfSize; fx++)
				for (int fy = 0; fy < rootRfSize; fy++) {
					int index = (rx + ry * rsc2d.getWidth()) + (fy + fx * rootRfSize) * rsc2d.getWidth() * rsc2d.getHeight();

					sf::Color color;
					color.r = (weightData[index] - minimum) * mult;
					color.g = color.r;
					color.b = color.r;
					color.a = 255;

					_image.setPixel(rx * rootRfSize + fx, ry * rootRfSize + fy, color);
				}
		}

	_texture.loadFromImage(_image);
}