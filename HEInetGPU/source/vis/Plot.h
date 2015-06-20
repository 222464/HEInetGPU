/*
HEInetGPU
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

namespace vis {
	struct Point {
		sf::Vector2f _position;

		sf::Color _color;

		Point()
			: _color(sf::Color::Black)
		{}
	};

	struct Curve {
		std::string _name;

		float _shadow;
		sf::Vector2f _shadowOffset;

		std::vector<Point> _points;

		Curve()
			: _shadow(0.5f), _shadowOffset(-4.0f, 4.0f)
		{}
	};

	struct Plot {
		sf::Color _axesColor;
		sf::Color _backgroundColor;

		std::vector<Curve> _curves;

		Plot()
			: _axesColor(sf::Color::Black), _backgroundColor(sf::Color::White)
		{}

		void draw(sf::RenderTarget &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
			const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements, float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision);
	};

	float vectorMagnitude(const sf::Vector2f &vector);
	sf::Vector2f vectorNormalize(const sf::Vector2f &vector);
	float vectorDot(const sf::Vector2f &left, const sf::Vector2f &right);
}