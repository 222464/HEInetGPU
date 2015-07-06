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

#include "Settings.h"

#if DEMO_SELECTION == DEMO_FEATURE_EXTRACTION

#include <system/ComputeSystem.h>

#include <ei/HEInet.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <array>

#include <time.h>
#include <iostream>

#include <random>

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;
	program.loadFromFile("resources/ei.cl", cs);

	std::shared_ptr<ei::EIlayer::Kernels> rsc2dKernels = std::make_shared<ei::EIlayer::Kernels>();

	rsc2dKernels->loadFromProgram(program);

	std::shared_ptr<ei::HEInet::Kernels> eiKernels = std::make_shared<ei::HEInet::Kernels>();

	eiKernels->loadFromProgram(program);

	ei::HEInet ht;

	sf::Image testImage;
	testImage.loadFromFile("testImage_whitened.png");

	int windowWidth = 16;
	int windowHeight = 16;

	std::vector<ei::EIlayer::Configuration> configs;

	std::vector<cl_int2> eSizes(1);
	std::vector<cl_int2> iSizes(1);

	eSizes[0].x = 16;
	eSizes[0].y = 16;
	//eSizes[1].x = 24;
	//eSizes[1].y = 24;
	//eSizes[2].x = 16;
	//eSizes[2].y = 16;

	iSizes[0].x = 8;
	iSizes[0].y = 8;
	//iSizes[1].x = 12;
	//iSizes[1].y = 12;
	//iSizes[2].x = 8;
	//iSizes[2].y = 8;

	cl_int2 inputSize = { windowWidth, windowHeight };

	ei::generateConfigsFromSizes(inputSize, eSizes, iSizes, configs);

	ht.createRandom(configs, 6, 6, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.02f, 0.02f, cs, rsc2dKernels, eiKernels, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), windowWidth, windowHeight);

	cl::Image2D zeroImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 1, 1);

	sf::RenderWindow window;

	sf::ContextSettings contextSettings;
	contextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(1280, 720), "HEInetGPU", sf::Style::Default, contextSettings);

	window.setFramerateLimit(60);

	std::uniform_int_distribution<int> distX(0, testImage.getSize().x - windowWidth - 1);
	std::uniform_int_distribution<int> distY(0, testImage.getSize().y - windowHeight - 1);

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

		cl::size_t<3> zeroCoord;
		zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

		cl::size_t<3> dims;
		dims[0] = windowWidth;
		dims[1] = windowHeight;
		dims[2] = 1;

		sf::Image subImage;

		subImage.create(windowWidth, windowHeight);

		subImage.copy(testImage, 0, 0, sf::IntRect(distX(generator), distY(generator), windowWidth, windowHeight));

		std::vector<float> inputData(windowWidth * windowHeight);

		for (int x = 0; x < windowWidth; x++)
			for (int y = 0; y < windowHeight; y++) {
				sf::Color c = subImage.getPixel(x, y);
				inputData[x + y * windowWidth] = (c.r + c.g + c.b) / (3.0f * 255.0f) * 0.5f; // Scale by 0.5f so maximum spike rate is 1/2 of the time
			}

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, zeroCoord, dims, 0, 0, inputData.data());

		ht.spikeSumBegin(cs);

		cl_uint4 zeroColor = { 0, 0, 0, 0 };

		ht.setInputPhase(cs, zeroColor);

		for (int iter = 0; iter < 50; iter++) {
			ht.update(cs, inputImage, zeroImage, 0.05f, 0.2f, 0.02f);
			ht.sumSpikes(cs, 1.0f / 50.0f);
			ht.learn(cs, zeroImage, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.02f, 0.02f);
			ht.stepEnd(cs);
		}

		ht.predict(cs);
		ht.learnPrediction(cs, inputImage, 0.005f);

		window.clear();

		std::vector<cl_float> iSpikeData(ht.getEIlayers()[0].getConfig()._iWidth * ht.getEIlayers()[0].getConfig()._iHeight);
		std::vector<cl_float> eSpikeData(ht.getEIlayers()[0].getConfig()._eWidth * ht.getEIlayers()[0].getConfig()._eHeight);
		std::vector<float> predictionData(windowWidth * windowHeight);

		cl::size_t<3> inputDims;
		inputDims[0] = windowWidth;
		inputDims[1] = windowHeight;
		inputDims[2] = 1;

		cl::size_t<3> eDims;
		eDims[0] = ht.getEIlayers()[0].getConfig()._eWidth;
		eDims[1] = ht.getEIlayers()[0].getConfig()._eHeight;
		eDims[2] = 1;

		cl::size_t<3> iDims;
		iDims[0] = ht.getEIlayers()[0].getConfig()._iWidth;
		iDims[1] = ht.getEIlayers()[0].getConfig()._iHeight;
		iDims[2] = 1;

		cs.getQueue().enqueueReadImage(ht._iSpikeSumsIterPrev, CL_TRUE, zeroCoord, iDims, 0, 0, iSpikeData.data());
		cs.getQueue().enqueueReadImage(ht._eSpikeSumsIterPrev, CL_TRUE, zeroCoord, eDims, 0, 0, eSpikeData.data());
		cs.getQueue().enqueueReadImage(ht._prediction, CL_TRUE, zeroCoord, inputDims, 0, 0, predictionData.data());

		{
			sf::Image img;
			img.create(iDims[0], iDims[1]);

			for (int x = 0; x < iDims[0]; x++)
				for (int y = 0; y < iDims[1]; y++) {
					sf::Color c;
					c.r = c.g = c.b = 255 * iSpikeData[x + y * iDims[0]];
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

		{
			sf::Image img;
			img.create(eDims[0], eDims[1]);

			for (int x = 0; x < eDims[0]; x++)
				for (int y = 0; y < eDims[1]; y++) {
					sf::Color c;
					c.r = c.g = c.b = 255 * eSpikeData[x + y * eDims[0]];
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
		}

		{
			cl::size_t<3> effWeightsDims;
			effWeightsDims[0] = ht.getEIlayers()[0].getConfig()._eWidth;
			effWeightsDims[1] = ht.getEIlayers()[0].getConfig()._eHeight;
			effWeightsDims[2] = std::pow(configs[0]._eFeedForwardRadius * 2 + 1, 2);

			std::vector<float> eWeights(eDims[0] * eDims[1] * effWeightsDims[2], 0.0f);

			cs.getQueue().enqueueReadImage(ht.getEIlayers()[0]._eFeedForwardWeights._weights, CL_TRUE, zeroCoord, effWeightsDims, 0, 0, eWeights.data());

			float mean = 0.5f;

			/*for (int i = 0; i < eWeights.size(); i++) {
				mean += eWeights[i];
			}

			mean /= eWeights.size();*/

			float variance = 0.0f;

			for (int i = 0; i < eWeights.size(); i++) {
				eWeights[i] -= mean;

				variance += eWeights[i] * eWeights[i];
			}

			variance /= eWeights.size();

			float stdDevInv = 1.0f / std::sqrt(variance);

			for (int i = 0; i < eWeights.size(); i++)
				eWeights[i] *= stdDevInv;

			sf::Image img;

			int diam = configs[0]._eFeedForwardRadius * 2 + 1;

			img.create(effWeightsDims[0] * diam, effWeightsDims[1] * diam);

			for (int rx = 0; rx < effWeightsDims[0]; rx++)
				for (int ry = 0; ry < effWeightsDims[1]; ry++) {
					for (int wx = 0; wx < diam; wx++)
						for (int wy = 0; wy < diam; wy++) {
							int index = (rx + ry * effWeightsDims[0]) + (wx + wy * diam) * effWeightsDims[0] * effWeightsDims[1];

							sf::Color c;
							c.r = c.g = c.b = 255 * (0.5f * eWeights[index] + 0.5f);
							c.a = 255;
							img.setPixel(rx * diam + wx, ry * diam + wy, c);
						}
				}

			sf::Texture tex;

			tex.loadFromImage(img);

			sf::Sprite s;

			s.setTexture(tex);
			s.setScale(2.0f, 2.0f);

			s.setPosition(0.0f, window.getSize().y - img.getSize().y * 2.0f);

			window.draw(s);
		}

		ht.predictionEnd();

		window.display();
	}

	return 0;
}

#endif