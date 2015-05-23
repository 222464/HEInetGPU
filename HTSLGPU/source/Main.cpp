#include <system/ComputeSystem.h>

#include <htsl/RecurrentSparseCoder2D.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>

#include <random>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;
	program.loadFromFile("resources/htsl.cl", cs);

	htsl::RecurrentSparseCoder2D rsc2d;

	rsc2d.createRandom(32, 32, 32, 32, 8, 8, 8, cs, program);

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "HTSLGPU", sf::Style::Default);

	bool quit = false;

	while (!quit) {
		sf::Event e;

		while (window.pollEvent(e)) {
			switch (e.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		window.clear();



		window.display();
	}

	return 0;
}