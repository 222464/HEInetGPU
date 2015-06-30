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

#if DEMO_SELECTION == DEMO_PREDICTION

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

	std::shared_ptr<ei::EIlayer::Kernels> layerKernels = std::make_shared<ei::EIlayer::Kernels>();

	layerKernels->loadFromProgram(program);

	std::shared_ptr<ei::HEInet::Kernels> hKernels = std::make_shared<ei::HEInet::Kernels>();

	hKernels->loadFromProgram(program);

	ei::HEInet ht;

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

	std::vector<ei::EIlayer::Configuration> configs;

	std::vector<cl_int2> eSizes(1);
	std::vector<cl_int2> iSizes(1);

	eSizes[0].x = 8;
	eSizes[0].y = 8;
	//eSizes[1].x = 12;
	//eSizes[1].y = 12;
	//eSizes[2].x = 8;
	//eSizes[2].y = 8;

	iSizes[0].x = 4;
	iSizes[0].y = 4;
	//iSizes[1].x = 6;
	//iSizes[1].y = 6;
	//iSizes[2].x = 4;
	//iSizes[2].y = 4;

	cl_int2 inputSize = { 2, 2 };

	ei::generateConfigsFromSizes(inputSize, eSizes, iSizes, configs);

	ht.createRandom(configs, 6, 6, -0.01f, 0.01f, 0.0f, 0.01f, 0.01f, 0.01f, 0.03f, 0.03f, cs, layerKernels, hKernels, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);

	cl::Image2D zeroImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 1, 1);

	sf::RenderWindow window;

	sf::ContextSettings contextSettings;
	contextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(1280, 720), "HEInetGPU", sf::Style::Default, contextSettings);

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
		dims[0] = inputSize.x;
		dims[1] = inputSize.y;
		dims[2] = 1;

		std::array<float, 4> normSequence;

		float mean = 0.0f;

		for (int i = 0; i < normSequence.size(); i++)
			mean += sequence[s][i];

		float variance = 0.0f;

		for (int i = 0; i < normSequence.size(); i++) {
			normSequence[i] = sequence[s][i] - mean;

			variance += normSequence[i] * normSequence[i];
		}

		float stdDevInv = 1.0f / std::sqrt(variance / normSequence.size());

		for (int i = 0; i < normSequence.size(); i++) {
			normSequence[i] *= stdDevInv;
		}

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, zeroCoord, dims, 0, 0, normSequence.data());

		ht.update(cs, inputImage, zeroImage, 30, 0.02f, 0.002f);
		ht.learn(cs, inputImage, zeroImage, 0.01f, 0.028f, 0.028f, 0.028f, 0.028f, 0.06f, 0.028f, 0.03f, 0.03f);
		//ht.learn(cs, inputImage, zeroImage, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.02f, 0.04f);
		ht.exStepEnd(cs);
		
		ht.predict(cs);
		ht.learnPrediction(cs, inputImage, 0.005f);

		window.clear();

		std::vector<cl_float> iSpikeData(ht.getEIlayers()[0].getConfig()._iWidth * ht.getEIlayers()[0].getConfig()._iHeight);
		std::vector<cl_float> eSpikeData(ht.getEIlayers()[0].getConfig()._eWidth * ht.getEIlayers()[0].getConfig()._eHeight);
		std::vector<float> predictionData(4);

		cl::size_t<3> inputDims;
		inputDims[0] = inputSize.x;
		inputDims[1] = inputSize.y;
		inputDims[2] = 1;

		cl::size_t<3> eDims;
		eDims[0] = ht.getEIlayers()[0].getConfig()._eWidth;
		eDims[1] = ht.getEIlayers()[0].getConfig()._eHeight;
		eDims[2] = 1;

		cl::size_t<3> iDims;
		iDims[0] = ht.getEIlayers()[0].getConfig()._iWidth;
		iDims[1] = ht.getEIlayers()[0].getConfig()._iHeight;
		iDims[2] = 1;

		cs.getQueue().enqueueReadImage(ht._iShortAveragePrevIter, CL_TRUE, zeroCoord, iDims, 0, 0, iSpikeData.data());
		cs.getQueue().enqueueReadImage(ht._eShortAveragePrevIter, CL_TRUE, zeroCoord, eDims, 0, 0, eSpikeData.data());
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

			sf::Image img;

			int diam = configs[0]._eFeedForwardRadius * 2 + 1;

			img.create(effWeightsDims[0] * diam, effWeightsDims[1] * diam);

			for (int rx = 0; rx < effWeightsDims[0]; rx++)
				for (int ry = 0; ry < effWeightsDims[1]; ry++) {
					for (int wx = 0; wx < diam; wx++)
						for (int wy = 0; wy < diam; wy++) {
							int index = (rx + ry * effWeightsDims[0]) + (wx + wy * diam) * effWeightsDims[0] * effWeightsDims[1];

							sf::Color c;
							c.r = c.g = c.b = 255 * sigmoid(4.0f * eWeights[index]);
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

		for (int i = 0; i < predictionData.size(); i++)
			std::cout << (predictionData[i] > 0.0f ? 1 : 0) << " ";

		std::cout << std::endl;

		window.display();
	}

	return 0;
}

#endif