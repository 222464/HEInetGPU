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

#include "../system/ComputeProgram.h"

#include <memory>
#include <random>

namespace htsl {
	class RecurrentSparseCoder2D : sys::Uncopyable {
	public:
		// Kernels this system uses
		struct Kernels {
			cl::Kernel _initializeKernel;
			cl::Kernel _excitationKernel;
			cl::Kernel _activateKernel;
			cl::Kernel _learnKernel;

			// Load kernels from program
			void loadFromProgram(sys::ComputeProgram &program);
		};

	private:
		std::shared_ptr<Kernels> _kernels;

		int _inputWidth, _inputHeight;
		int _width, _height;
		int _receptiveRadius, _recurrentRadius, _inhibitionRadius;

	public:
		// Images
		cl::Image2D _excitations;

		cl::Image2D _activations;
		cl::Image2D _activationsPrev;

		cl::Image2D _spikes;
		cl::Image2D _spikesPrev;
		cl::Image2D _spikesRecurrentPrev;

		cl::Image2D _states;
		cl::Image2D _statesPrev;

		cl::Image3D _hiddenVisibleWeights;
		cl::Image3D _hiddenVisibleWeightsPrev;
		cl::Image3D _hiddenHiddenPrevWeights;
		cl::Image3D _hiddenHiddenPrevWeightsPrev;
		cl::Image3D _hiddenHiddenWeights;
		cl::Image3D _hiddenHiddenWeightsPrev;
		cl::Image2D _biases;
		cl::Image2D _biasesPrev;

		// Create with random weights
		void createRandom(int inputWidth, int inputHeight, int width, int height,
			int receptiveRadius, int recurrentRadius, int inhibitionRadius, float ffWeight, float lWeight, float initBias,
			sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &kernels, std::mt19937 &generator);

		// Find sparse codes
		void update(sys::ComputeSystem &cs, const cl::Image2D &inputs, float dt, int iterations = 50);

		// Learn sparse codes
		void learn(sys::ComputeSystem &cs, const cl::Image2D &inputs, float alpha, float beta, float gamma, float delta, float sparsity, int iterations = 50);
		
		// Swap recurrent input buffers
		void stepEnd();

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		int getWidth() const {
			return _width;
		}

		int getHeight() const {
			return _height;
		}

		int getReceptiveRadius() const {
			return _receptiveRadius;
		}

		int getRecurrentRadius() const {
			return _recurrentRadius;
		}

		int getInhibitionRadius() const {
			return _inhibitionRadius;
		}

		const std::shared_ptr<Kernels> &getKernels() const {
			return _kernels;
		}
	};
}