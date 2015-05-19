#include "../system/ComputeProgram.h"

namespace htsl {
	class RecurrentSparseCoder2D {
	private:
		cl::Image2D _activations;

		cl::Image2D _states;
		cl::Image2D _statesPrev;

		cl::Image2D _reconstruction;

		cl::Image3D _hiddenVisibleWeights;
		cl::Image3D _hiddenVisibleWeightsPrev;
		cl::Image3D _hiddenHiddenPrevWeights;
		cl::Image3D _hiddenHiddenPrevWeightsPrev;
		cl::Image3D _hiddenHiddenWeights;
		cl::Image3D _hiddenHiddenWeightsPrev;
		cl::Image2D _biases;
		cl::Image2D _biasesPrev;

		cl::Kernel _activateKernel;
		cl::Kernel _reconstructKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _learnKernel;

		int _inputWidth, _inputHeight;
		int _width, _height;
		int _receptiveRadius, _recurrentRadius, _inhibitionRadius;

	public:
		void createRandom(int inputWidth, int inputHeight, int width, int height,
			int receptiveRadius, int recurrentRadius, int inhibitionRadius,
			sys::ComputeSystem &cs, sys::ComputeProgram &program);

		void update(sys::ComputeSystem &cs, const cl::Image2D &input);
		void learn(sys::ComputeSystem &cs, const cl::Image2D &input, float alpha, float beta, float gamma, float delta);
		void stepEnd();
	};
}