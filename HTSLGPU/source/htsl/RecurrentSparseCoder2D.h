#pragma once

#include "../system/ComputeProgram.h"

#include <memory>

namespace htsl {
	class RecurrentSparseCoder2D {
	private:
		struct Kernels {
			cl::Kernel _initializeKernel;
			cl::Kernel _activateKernel;
			cl::Kernel _reconstructReceptiveKernel;
			cl::Kernel _reconstructRecurrentKernel;
			cl::Kernel _errorKernel;
			cl::Kernel _inhibitKernel;
			cl::Kernel _learnKernel;

			void loadFromProgram(sys::ComputeProgram &program);
		};

		std::shared_ptr<Kernels> _kernels;

		int _inputWidth, _inputHeight;
		int _width, _height;
		int _receptiveRadius, _recurrentRadius, _inhibitionRadius;

	public:
		cl::Image2D _activations;
		cl::Image2D _inhibitions;

		cl::Image2D _states;
		cl::Image2D _statesPrev;

		cl::Image2D _receptiveReconstruction;
		cl::Image2D _recurrentReconstruction;
		cl::Image2D _receptiveErrors;
		cl::Image2D _recurrentErrors;

		cl::Image3D _hiddenVisibleWeights;
		cl::Image3D _hiddenVisibleWeightsPrev;
		cl::Image3D _hiddenHiddenPrevWeights;
		cl::Image3D _hiddenHiddenPrevWeightsPrev;
		cl::Image3D _hiddenHiddenWeights;
		cl::Image3D _hiddenHiddenWeightsPrev;
		cl::Image2D _biases;
		cl::Image2D _biasesPrev;

		void createRandom(int inputWidth, int inputHeight, int width, int height,
			int receptiveRadius, int recurrentRadius, int inhibitionRadius,
			sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &kernels);

		void update(sys::ComputeSystem &cs, const cl::Image2D &inputs);
		void learn(sys::ComputeSystem &cs, const cl::Image2D &inputs, float alpha, float beta, float gamma, float delta, float sparsity);
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