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

#include <system/ComputeSystem.h>

#include <htsl/RecurrentSparseCoder2D.h>
#include <vis/ReceptiveFields.h>
#include <vis/PrettySDR.h>

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

	std::shared_ptr<htsl::RecurrentSparseCoder2D::Kernels> rsc2dKernels = std::make_shared<htsl::RecurrentSparseCoder2D::Kernels>();

	rsc2dKernels->loadFromProgram(program);

	htsl::RecurrentSparseCoder2D rsc2d;

	sf::Image testImage;
	testImage.loadFromFile("testImage.png");

	int windowWidth = 32;
	int windowHeight = 32;

	rsc2d.createRandom(windowWidth, windowHeight, 32, 32, 6, 4, 4, 0.01f, 0.0f, 0.5f, cs, rsc2dKernels, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), windowWidth, windowHeight);

	vis::ReceptiveFields rfs;
	rfs.create(rsc2d);

	vis::PrettySDR psdr;

	psdr.create(rsc2d.getWidth(), rsc2d.getHeight());

	std::uniform_int_distribution<int> distSampleX(0, testImage.getSize().x - windowWidth - 1);
	std::uniform_int_distribution<int> distSampleY(0, testImage.getSize().y - windowHeight - 1);

	sf::RenderWindow window;

	sf::ContextSettings contextSettings;
	contextSettings.antialiasingLevel = 8;

	window.create(sf::VideoMode(1280, 720), "HTSLGPU", sf::Style::Default, contextSettings);

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

		int sx = distSampleX(generator);
		int sy = distSampleY(generator);

		std::vector<float> imageData(windowWidth * windowHeight);

		for (int wx = 0; wx < windowWidth; wx++)
			for (int wy = 0; wy < windowHeight; wy++) {
				int x = sx + wx;
				int y = sy + wy;

				sf::Color color = testImage.getPixel(x, y);

				imageData[wx + wy * windowWidth] = (color.r * 0.333f + color.g * 0.333f + color.b * 0.333f) / 255.0f;
			}

		cl::size_t<3> zeroCoord;
		zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

		cl::size_t<3> dims;
		dims[0] = windowWidth;
		dims[1] = windowHeight;
		dims[2] = 1;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, zeroCoord, dims, 0, 0, imageData.data());

		rsc2d.update(cs, inputImage, 0.1f);
		rsc2d.learn(cs, inputImage, 0.005f, 0.005f, 1.0f, 0.1f, 0.05f);
		rsc2d.stepEnd();

		rfs.render(rsc2d, cs);

		psdr.loadFromImage(cs, rsc2d);

		window.clear();

		sf::Sprite rfsSprite;
		rfsSprite.setTexture(rfs.getTexture());

		window.draw(rfsSprite);

		psdr.render(window, sf::Vector2f(window.getSize().x - rsc2d.getWidth() * psdr._nodeSpaceSize, 0.0f));

		window.display();
	}

	return 0;
}