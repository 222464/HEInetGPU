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

#pragma once

#include <SFML/Graphics.hpp>

#include "../htsl/RecurrentSparseCoder2D.h"

namespace vis {
	class PrettySDR {
	private:
		std::vector<float> _nodes;

		int _width, _height;

	public:
		float _edgeRadius;
		float _nodeSpaceSize;
		float _nodeOuterRatio;
		float _nodeInnerRatio;
		
		int _edgeSegments;
		int _nodeOuterSegments;
		int _nodeInnerSegments;

		sf::Color _backgroundColor;
		sf::Color _nodeOuterColor;
		sf::Color _nodeInnerColor;

		PrettySDR()
			: _edgeRadius(4.0f), _nodeSpaceSize(16.0f), _nodeOuterRatio(0.85f), _nodeInnerRatio(0.75f),
			_edgeSegments(16), _nodeOuterSegments(16), _nodeInnerSegments(16),
			_backgroundColor(128, 128, 128), _nodeOuterColor(64, 64, 64), _nodeInnerColor(255, 0, 0)
		{}

		void create(int width, int height);

		float &operator[](int index) {
			return _nodes[index];
		}

		float &at(int x, int y) {
			return _nodes[x + y * _width];
		}

		void loadFromImage(sys::ComputeSystem &cs, const htsl::RecurrentSparseCoder2D &rsc2d);

		void render(sf::RenderTarget &rt, const sf::Vector2f &position);
	};
}