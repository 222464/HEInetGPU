#pragma once

#include <SFML/Graphics.hpp>

#include "../htsl/RecurrentSparseCoder2D.h"

namespace vis {
	class ReceptiveFields {
	private:
		sf::Image _image;
		sf::Texture _texture;

	public:
		sf::Color _outlineColor;
		float _outlineWidth;

		ReceptiveFields()
			: _outlineColor(sf::Color::Red), _outlineWidth(1.0f)
		{}

		void create(const htsl::RecurrentSparseCoder2D &rsc2d);

		void render(const htsl::RecurrentSparseCoder2D &rsc2d, sys::ComputeSystem &cs);

		const sf::Image &getImage() const {
			return _image;
		}

		const sf::Texture &getTexture() const {
			return _texture;
		}
	};
}