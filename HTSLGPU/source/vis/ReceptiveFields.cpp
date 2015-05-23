#include "ReceptiveFields.h"

using namespace vis;

void ReceptiveFields::create(const htsl::RecurrentSparseCoder2D &rsc2d) {
	int rfSize = rsc2d.getReceptiveRadius() * 2 + 1;

	_image.create(rsc2d.getWidth() * rfSize, rsc2d.getHeight() * rfSize);
}

void ReceptiveFields::render(const htsl::RecurrentSparseCoder2D &rsc2d, sys::ComputeSystem &cs) {
	int rfSize = rsc2d.getReceptiveRadius() * 2 + 1;

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> weightDims;
	weightDims[0] = rsc2d.getWidth();
	weightDims[1] = rsc2d.getHeight();
	weightDims[2] = rfSize;

	std::vector<cl_float2> weightData(weightDims[0] * weightDims[1] * weightDims[2]);

	cs.getQueue().enqueueReadImage(rsc2d._hiddenVisibleWeights, CL_TRUE, zeroCoord, weightDims, 0, 0, weightData.data());

	for (int rx = 0; rx < rsc2d.getWidth(); rx++)
		for (int ry = 0; ry < rsc2d.getHeight(); ry++) {
			for (int fx = 0; fx < rfSize; fx++)
				for (int fy = 0; fy < rfSize; fy++) {
					int index = rx + ry * rsc2d.getWidth() * rfSize + (fx + fy * rfSize) * rsc2d.getWidth() * rsc2d.getHeight();

					sf::Color color;
					color.r = weightData[index].x * 255;
					color.g = weightData[index].y * 255;
					color.b = 255;
					color.a = 255;

					_image.setPixel(rx * rfSize + fx, ry * rfSize + fy, color);
				}
		}

	_texture.loadFromImage(_image);
}