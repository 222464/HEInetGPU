#include "RecurrentSparseCoder2D.h"

using namespace htsl;

void RecurrentSparseCoder2D::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	_initializeKernel = cl::Kernel(program.getProgram(), "rscInitialize");
	_activateKernel = cl::Kernel(program.getProgram(), "rscActivate");
	_reconstructReceptiveKernel = cl::Kernel(program.getProgram(), "rscReconstructReceptive");
	_reconstructRecurrentKernel = cl::Kernel(program.getProgram(), "rscReconstructRecurrent");
	_errorKernel = cl::Kernel(program.getProgram(), "rscError");
	_inhibitKernel = cl::Kernel(program.getProgram(), "rscInhibit");
	_learnKernel = cl::Kernel(program.getProgram(), "rscLearn");
}

void RecurrentSparseCoder2D::createRandom(int inputWidth, int inputHeight, int width, int height,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius,
	sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &kernels, std::mt19937 &generator)
{
	_kernels = kernels;

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	_width = width;
	_height = height;

	_receptiveRadius = receptiveRadius;
	_recurrentRadius = recurrentRadius;
	_inhibitionRadius = inhibitionRadius;

	int receptiveSize = std::pow(_receptiveRadius * 2 + 1, 2);
	int recurrentSize = std::pow(_recurrentRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(_inhibitionRadius * 2 + 1, 2);

	_activations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_inhibitions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_states = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_statesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_receptiveReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_receptiveErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_hiddenVisibleWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, receptiveSize);
	_hiddenVisibleWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, receptiveSize);

	_hiddenHiddenPrevWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, recurrentSize);
	_hiddenHiddenPrevWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, recurrentSize);

	_hiddenHiddenWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionSize);
	_hiddenHiddenWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionSize);
	
	_biases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_biasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dims;
	dims[0] = _width;
	dims[1] = _height;
	dims[2] = 1;

	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dims);
	cs.getQueue().enqueueFillImage(_biasesPrev, zeroColor, zeroCoord, dims);

	int index = 0;

	std::uniform_int_distribution<int> seedDist(0, 10000);

	cl_uint2 seed = { seedDist(generator), seedDist(generator) };

	_kernels->_initializeKernel.setArg(index++, _hiddenVisibleWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, receptiveSize);
	_kernels->_initializeKernel.setArg(index++, recurrentSize);
	_kernels->_initializeKernel.setArg(index++, inhibitionSize);
	_kernels->_initializeKernel.setArg(index++, seed);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_initializeKernel, cl::NullRange, cl::NDRange(_width, _height));	
}

void RecurrentSparseCoder2D::update(sys::ComputeSystem &cs, const cl::Image2D &inputs) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height - 1) };
	cl_float2 inputDimsToDims = { static_cast<float>(_width + 1) / static_cast<float>(_inputWidth + 1), static_cast<float>(_height + 1) / static_cast<float>(_inputHeight + 1) };

	cl_int2 reverseReceptiveRadii = { std::ceil(_receptiveRadius * inputDimsToDims.x), std::ceil(_receptiveRadius * inputDimsToDims.y) };

	// Activate
	{
		int index = 0;

		_kernels->_activateKernel.setArg(index++, inputs);
		_kernels->_activateKernel.setArg(index++, _statesPrev);
		_kernels->_activateKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_kernels->_activateKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_kernels->_activateKernel.setArg(index++, _activations);
		_kernels->_activateKernel.setArg(index++, inputDims);
		_kernels->_activateKernel.setArg(index++, dims);
		_kernels->_activateKernel.setArg(index++, dimsToInputDims);
		_kernels->_activateKernel.setArg(index++, _receptiveRadius);
		_kernels->_activateKernel.setArg(index++, _recurrentRadius);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_activateKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Inhibit
	{
		int index = 0;

		_kernels->_inhibitKernel.setArg(index++, _activations);
		_kernels->_inhibitKernel.setArg(index++, _hiddenHiddenWeightsPrev);
		_kernels->_inhibitKernel.setArg(index++, _biasesPrev);
		_kernels->_inhibitKernel.setArg(index++, _states);
		_kernels->_inhibitKernel.setArg(index++, _inhibitions);
		_kernels->_inhibitKernel.setArg(index++, dims);
		_kernels->_inhibitKernel.setArg(index++, _inhibitionRadius);
		_kernels->_inhibitKernel.setArg(index++, 1.0f / _inhibitionRadius);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_inhibitKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Reconstruction - receptive
	{
		int index = 0;

		_kernels->_reconstructReceptiveKernel.setArg(index++, _states);
		_kernels->_reconstructReceptiveKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_kernels->_reconstructReceptiveKernel.setArg(index++, _receptiveReconstruction);
		_kernels->_reconstructReceptiveKernel.setArg(index++, inputDims);
		_kernels->_reconstructReceptiveKernel.setArg(index++, dims);
		_kernels->_reconstructReceptiveKernel.setArg(index++, dimsToInputDims);
		_kernels->_reconstructReceptiveKernel.setArg(index++, inputDimsToDims);
		_kernels->_reconstructReceptiveKernel.setArg(index++, _receptiveRadius);
		_kernels->_reconstructReceptiveKernel.setArg(index++, reverseReceptiveRadii);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_reconstructReceptiveKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
	}

	// Reconstruction - recurrent
	{
		int index = 0;

		_kernels->_reconstructRecurrentKernel.setArg(index++, _states);
		_kernels->_reconstructRecurrentKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_kernels->_reconstructRecurrentKernel.setArg(index++, _recurrentReconstruction);
		_kernels->_reconstructRecurrentKernel.setArg(index++, dims);
		_kernels->_reconstructRecurrentKernel.setArg(index++, _recurrentRadius);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_reconstructRecurrentKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Error - receptive
	{
		int index = 0;

		_kernels->_errorKernel.setArg(index++, inputs);
		_kernels->_errorKernel.setArg(index++, _receptiveReconstruction);
		_kernels->_errorKernel.setArg(index++, _receptiveErrors);
		
		cs.getQueue().enqueueNDRangeKernel(_kernels->_errorKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
	}

	// Error - recurrent
	{
		int index = 0;

		_kernels->_errorKernel.setArg(index++, _statesPrev);
		_kernels->_errorKernel.setArg(index++, _recurrentReconstruction);
		_kernels->_errorKernel.setArg(index++, _recurrentErrors);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_errorKernel, cl::NullRange, cl::NDRange(_width, _height));
	}
}

void RecurrentSparseCoder2D::learn(sys::ComputeSystem &cs, const cl::Image2D &inputs, float alpha, float beta, float gamma, float delta, float sparsity) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height - 1) };
	cl_float4 learningRates = { alpha, beta, gamma, delta };

	// Learn
	{
		int index = 0;

		_kernels->_learnKernel.setArg(index++, _receptiveErrors);
		_kernels->_learnKernel.setArg(index++, _recurrentErrors);
		_kernels->_learnKernel.setArg(index++, _activations);
		_kernels->_learnKernel.setArg(index++, _inhibitions);
		_kernels->_learnKernel.setArg(index++, _states);
		_kernels->_learnKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _biasesPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenVisibleWeights);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenPrevWeights);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenWeights);
		_kernels->_learnKernel.setArg(index++, _biases);
		_kernels->_learnKernel.setArg(index++, inputDims);
		_kernels->_learnKernel.setArg(index++, dims);
		_kernels->_learnKernel.setArg(index++, dimsToInputDims);
		_kernels->_learnKernel.setArg(index++, _receptiveRadius);
		_kernels->_learnKernel.setArg(index++, _recurrentRadius);
		_kernels->_learnKernel.setArg(index++, _inhibitionRadius);
		_kernels->_learnKernel.setArg(index++, learningRates);
		_kernels->_learnKernel.setArg(index++, sparsity);
		_kernels->_learnKernel.setArg(index++, sparsity * sparsity);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_learnKernel, cl::NullRange, cl::NDRange(_width, _height));
	}
}

void RecurrentSparseCoder2D::stepEnd() {
	std::swap(_states, _statesPrev);
	std::swap(_hiddenVisibleWeights, _hiddenVisibleWeightsPrev);
	std::swap(_hiddenHiddenPrevWeights, _hiddenHiddenPrevWeightsPrev);
	std::swap(_hiddenHiddenWeights, _hiddenHiddenWeightsPrev);
	std::swap(_biases, _biasesPrev);
}