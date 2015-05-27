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

#include "PrettySDR.h"

using namespace vis;

void PrettySDR::create(int width, int height) {
	_width = width;
	_height = height;

	_nodes.clear();
	_nodes.assign(width * height, 0.0f);
}

void PrettySDR::loadFromImage(sys::ComputeSystem &cs, const htsl::RecurrentSparseCoder2D &rsc2d) {
	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> sdrDims;
	sdrDims[0] = rsc2d.getWidth();
	sdrDims[1] = rsc2d.getHeight();
	sdrDims[2] = 1;

	cs.getQueue().enqueueReadImage(rsc2d._spikes, CL_TRUE, zeroCoord, sdrDims, 0, 0, _nodes.data());
}

void PrettySDR::render(sf::RenderTarget &rt, const sf::Vector2f &position) {
	float rWidth = _nodeSpaceSize * _width;
	float rHeight = _nodeSpaceSize * _height;

	sf::RectangleShape rsHorizontal;
	rsHorizontal.setPosition(position + sf::Vector2f(0.0f, _edgeRadius));
	rsHorizontal.setSize(sf::Vector2f(rWidth, rHeight - _edgeRadius * 2.0f));
	rsHorizontal.setFillColor(_backgroundColor);
	
	rt.draw(rsHorizontal);

	sf::RectangleShape rsVertical;
	rsVertical.setPosition(position + sf::Vector2f(_edgeRadius, 0.0f));
	rsVertical.setSize(sf::Vector2f(rWidth - _edgeRadius * 2.0f, rHeight));
	rsVertical.setFillColor(_backgroundColor);

	rt.draw(rsVertical);

	// Corners
	sf::CircleShape corner;
	corner.setRadius(_edgeRadius);
	corner.setPointCount(_edgeSegments);
	corner.setFillColor(_backgroundColor);
	corner.setOrigin(sf::Vector2f(_edgeRadius, _edgeRadius));

	corner.setPosition(position + sf::Vector2f(_edgeRadius, _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(rWidth - _edgeRadius, _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(rWidth - _edgeRadius, rHeight - _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(_edgeRadius, rHeight - _edgeRadius));
	rt.draw(corner);

	// Nodes
	sf::CircleShape outer;
	outer.setRadius(_nodeSpaceSize * _nodeOuterRatio * 0.5f);
	outer.setPointCount(_nodeOuterSegments);
	outer.setFillColor(_nodeOuterColor);
	outer.setOrigin(sf::Vector2f(outer.getRadius(), outer.getRadius()));

	sf::CircleShape inner;
	inner.setRadius(_nodeSpaceSize * _nodeOuterRatio * _nodeInnerRatio * 0.5f);
	inner.setPointCount(_nodeInnerSegments);
	inner.setFillColor(_nodeInnerColor);
	inner.setOrigin(sf::Vector2f(inner.getRadius(), inner.getRadius()));

	for (int x = 0; x < _width; x++)
		for (int y = 0; y < _height; y++) {
			outer.setPosition(position + sf::Vector2f((x + 0.5f) * _nodeSpaceSize, (y + 0.5f) * _nodeSpaceSize));

			rt.draw(outer);

			inner.setPosition(position + sf::Vector2f((x + 0.5f) * _nodeSpaceSize, (y + 0.5f) * _nodeSpaceSize));

			inner.setFillColor(sf::Color(_nodeInnerColor.r, _nodeInnerColor.g, _nodeInnerColor.b, 255 * at(x, y)));

			rt.draw(inner);
		}
}