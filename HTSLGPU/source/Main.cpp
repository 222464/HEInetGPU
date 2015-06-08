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

#include <htsl/HTSL.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <array>

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

	std::shared_ptr<htsl::HTSL::Kernels> htslKernels = std::make_shared<htsl::HTSL::Kernels>();

	htslKernels->loadFromProgram(program);

	htsl::HTSL ht;

	sf::Image testImage;
	testImage.loadFromFile("testImageWhitened.png");

	int windowWidth = 2;
	int windowHeight = 2;

	float sequence[8][4] = {
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 0.0f }
	};

	std::vector<htsl::RecurrentSparseCoder2D::Configuration> configs;

	std::vector<cl_int2> eSizes(3);
	std::vector<cl_int2> iSizes(3);

	eSizes[0].x = 32;
	eSizes[0].y = 32;
	eSizes[1].x = 24;
	eSizes[1].y = 24;
	eSizes[2].x = 16;
	eSizes[2].y = 16;

	iSizes[0].x = 16;
	iSizes[0].y = 16;
	iSizes[1].x = 12;
	iSizes[1].y = 12;
	iSizes[2].x = 8;
	iSizes[2].y = 8;

	cl_int2 inputSize = { windowWidth, windowHeight };

	htsl::generateConfigsFromSizes(inputSize, eSizes, iSizes, configs);

	ht.createRandom(configs, 6, 6, -0.05f, 0.05f, 0.0f, 0.05f, 0.01f, 0.01f, cs, rsc2dKernels, htslKernels, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), windowWidth, windowHeight);

	cl::Image2D zeroImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 1, 1);

	std::uniform_int_distribution<int> distSampleX(0, testImage.getSize().x - windowWidth - 1);
	std::uniform_int_distribution<int> distSampleY(0, testImage.getSize().y - windowHeight - 1);

	sf::RenderWindow window;

	sf::ContextSettings contextSettings;
	contextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(1280, 720), "HTSLGPU", sf::Style::Default, contextSettings);

	window.setFramerateLimit(60);

	bool quit = false;

	int s = 0;

	while (!quit) {
		sf::Event e;

		while (window.pollEvent(e)) {
			switch (e.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		s = (s + 1) % 8;

		if (s == 0) {
			std::cout << "Sequence:" << std::endl;
		}

		cl::size_t<3> zeroCoord;
		zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

		cl::size_t<3> dims;
		dims[0] = windowWidth;
		dims[1] = windowHeight;
		dims[2] = 1;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, zeroCoord, dims, 0, 0, sequence[s]);

		for (int iter = 0; iter < 17; iter++) {
			ht.update(cs, inputImage, zeroImage, 0.1f, 0.01f);
			ht.learn(cs, inputImage, zeroImage, 0.01f, 0.01f, 0.005f, 0.01f, 0.01f, 0.01f, 0.005f, 0.05f);
			ht.stepEnd();
		}
		
		ht.predict(cs);
		ht.learnPrediction(cs, inputImage, 0.01f);
		ht.predictionEnd(cs);

		window.clear();

		std::vector<cl_float2> iSpikeData(ht.getRSCLayers()[0].getConfig()._iWidth * ht.getRSCLayers()[0].getConfig()._iHeight);
		std::vector<float> predictionData(windowWidth * windowHeight);

		cl::size_t<3> inputDims;
		inputDims[0] = windowWidth;
		inputDims[1] = windowHeight;
		inputDims[2] = 1;

		cl::size_t<3> eDims;
		eDims[0] = ht.getRSCLayers()[0].getConfig()._eWidth;
		eDims[1] = ht.getRSCLayers()[0].getConfig()._eHeight;
		eDims[2] = 1;

		cl::size_t<3> iDims;
		iDims[0] = ht.getRSCLayers()[0].getConfig()._iWidth;
		iDims[1] = ht.getRSCLayers()[0].getConfig()._iHeight;
		iDims[2] = 1;

		cs.getQueue().enqueueReadImage(ht.getRSCLayers()[0]._eLayer._states, CL_TRUE, zeroCoord, iDims, 0, 0, iSpikeData.data());
		cs.getQueue().enqueueReadImage(ht._prediction, CL_TRUE, zeroCoord, inputDims, 0, 0, predictionData.data());

		{
			sf::Image img;
			img.create(iDims[0], iDims[1]);

			for (int x = 0; x < iDims[0]; x++)
				for (int y = 0; y < iDims[1]; y++) {
					sf::Color c;
					c.r = c.g = c.b = 255 * iSpikeData[x + y * iDims[0]].x;
					c.a = 255;

					img.setPixel(x, y, c);
				}

			sf::Texture tex;

			tex.loadFromImage(img);

			sf::Sprite s;
			s.setTexture(tex);

			s.setScale(4.0f, 4.0f);

			window.draw(s);
		}

		/*{
			sf::Image img;
			img.create(inputDims[0], inputDims[1]);

			for (int x = 0; x < inputDims[0]; x++)
				for (int y = 0; y < inputDims[1]; y++) {
					sf::Color c;
					c.r = c.g = c.b = 255 * std::min<float>(1.0f, std::max<float>(0.0f, predictionData[x + y * inputDims[0]]));
					c.a = 255;

					img.setPixel(x, y, c);
				}

			sf::Texture tex;

			tex.loadFromImage(img);

			sf::Sprite s;
			s.setTexture(tex);

			s.setPosition(8.0f * iDims[0], 0.0f);

			s.setScale(4.0f, 4.0f);

			window.draw(s);
		}*/

		for (int i = 0; i < predictionData.size(); i++)
			std::cout << (predictionData[i] > 0.5f ? 1 : 0) << " ";

		std::cout << std::endl;

		window.display();
	}

	return 0;
}