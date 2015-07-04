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

#include "../system/ComputeProgram.h"

#include <memory>
#include <random>

namespace ei {
	class EIlayer {
	public:
		// Kernels this system uses
		struct Kernels {
			cl::Kernel _eInitializeKernel;
			cl::Kernel _iInitializeKernel;

			cl::Kernel _eActivationKernel;
			cl::Kernel _iActivationKernel;

			cl::Kernel _eLearnKernel;
			cl::Kernel _iLearnKernel;

			// Load kernels from program
			void loadFromProgram(sys::ComputeProgram &program);
		};

		struct NeuronLayer {
			cl::Image2D _activations;
			cl::Image2D _activationsPrev;

			cl::Image2D _states;
			cl::Image2D _statesPrev;

			cl::Image2D _statesHistory;
			cl::Image2D _statesHistoryPrev;

			cl::Image2D _thresholds;
			cl::Image2D _thresholdsPrev;

			cl::Image2D _currentAverages;
			cl::Image2D _currentAveragesPrev;
		};

		struct Weights2D {
			cl::Image3D _weights;
			cl::Image3D _weightsPrev;
		};

		struct Configuration {
			int _eFeedForwardWidth, _eFeedForwardHeight;
			int _eWidth, _eHeight;
			int _iWidth, _iHeight;
			int _iFeedBackWidth, _iFeedBackHeight;
			int _eFeedForwardRadius;
			int _eFeedBackRadius;
			int _iFeedForwardRadius;
			int _iLateralRadius;
			int _iFeedBackRadius;

			Configuration()
				: _eFeedForwardWidth(8), _eFeedForwardHeight(8),
				_eWidth(16), _eHeight(16),
				_iWidth(8), _iHeight(8),
				_iFeedBackWidth(8), _iFeedBackHeight(8),
				_eFeedForwardRadius(8),
				_eFeedBackRadius(6),
				_iFeedForwardRadius(6),
				_iLateralRadius(6),
				_iFeedBackRadius(6)
			{}
		};

	private:
		std::shared_ptr<Kernels> _kernels;

		Configuration _config;

	public:
		// Image sets
		NeuronLayer _eLayer;
		NeuronLayer _iLayer;

		Weights2D _eFeedForwardWeights;
		Weights2D _eFeedBackWeights;
		Weights2D _iFeedForwardWeights;
		Weights2D _iLateralWeights;
		Weights2D _iFeedBackWeights;

		// Create with random weights
		void createRandom(const Configuration &config,
			float minInitEWeight, float maxInitEWeight,
			float minInitIWeight, float maxInitIWeight,
			float initEThreshold, float initIThreshold,
			float sparsityE, float sparsityI,
			sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &eilKernels, std::mt19937 &generator);

		// Find sparse codes
		void eActivate(sys::ComputeSystem &cs, const cl::Image2D &feedForwardInputs, float eta, float shDecay, float caDecay);
		void iActivate(sys::ComputeSystem &cs, const cl::Image2D &feedBackInputs, float eta, float shDecay, float caDecay);

		// Learn sparse codes
		void learn(sys::ComputeSystem &cs,
			const cl::Image2D &feedForwardInputs, const cl::Image2D &feedForwardInputsPrev,
			const cl::Image2D &feedBackInputs, const cl::Image2D &feedBackInputsPrev,
			float eAlpha, float eBeta, float eDelta,
			float iAlpha, float iBeta, float iGamma, float iDelta,
			float sparsityE, float sparsityI);

		// End of simulation step
		void stepEnd();

		const std::shared_ptr<Kernels> &getKernels() const {
			return _kernels;
		}

		const Configuration &getConfig() const {
			return _config;
		}
	};
}