#include "RecurrentSparseCoder2D.h"

using namespace htsl;

void RecurrentSparseCoder2D::createRandom(int inputWidth, int inputHeight, int width, int height,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius,
	sys::ComputeSystem &cs, sys::ComputeProgram &program)
{
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

	_receptiveReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_receptiveErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_hiddenVisibleWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, receptiveRadius);
	_hiddenVisibleWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, receptiveRadius);

	_hiddenHiddenPrevWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, recurrentSize);
	_hiddenHiddenPrevWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _width, _height, recurrentSize);

	_hiddenHiddenWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionRadius);
	_hiddenHiddenWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionRadius);
	
	_biases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_biasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	cl_uint4 zeroColor = { 0, 0, 0, 0 };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dims;
	dims[0] = _width;
	dims[1] = _height;
	dims[2] = 1;

	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dims);
	cs.getQueue().enqueueFillImage(_biasesPrev, zeroColor, zeroCoord, dims);

	cl::Kernel initializeKernel(program.getProgram(), "rscInitialize");

	int index = 0;

	initializeKernel.setArg(index++, _hiddenVisibleWeightsPrev);
	initializeKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
	initializeKernel.setArg(index++, _hiddenHiddenWeightsPrev);
	initializeKernel.setArg(index++, receptiveSize);
	initializeKernel.setArg(index++, recurrentSize);
	initializeKernel.setArg(index++, inhibitionSize);

	cs.getQueue().enqueueNDRangeKernel(initializeKernel, cl::NullRange, cl::NDRange(_width, _height));

	_activateKernel = cl::Kernel(program.getProgram(), "rscActivate");
	_reconstructReceptiveKernel = cl::Kernel(program.getProgram(), "rscReconstructReceptive");
	_reconstructRecurrentKernel = cl::Kernel(program.getProgram(), "rscReconstructRecurrent");
	_errorKernel = cl::Kernel(program.getProgram(), "rscError");
	_inhibitKernel = cl::Kernel(program.getProgram(), "rscInhibit");
	_learnKernel = cl::Kernel(program.getProgram(), "rscLearn");
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

		_activateKernel.setArg(index++, inputs);
		_activateKernel.setArg(index++, _statesPrev);
		_activateKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_activateKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_activateKernel.setArg(index++, _activations);
		_activateKernel.setArg(index++, inputDims);
		_activateKernel.setArg(index++, dims);
		_activateKernel.setArg(index++, dimsToInputDims);
		_activateKernel.setArg(index++, _receptiveRadius);
		_activateKernel.setArg(index++, _recurrentRadius);

		cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Inhibit
	{
		int index = 0;

		_inhibitKernel.setArg(index++, _activations);
		_inhibitKernel.setArg(index++, _hiddenHiddenWeightsPrev);
		_inhibitKernel.setArg(index++, _biasesPrev);
		_inhibitKernel.setArg(index++, _states);
		_inhibitKernel.setArg(index++, _inhibitions);
		_inhibitKernel.setArg(index++, dims);
		_inhibitKernel.setArg(index++, _inhibitionRadius);

		cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Reconstruction - receptive
	{
		int index = 0;

		_reconstructReceptiveKernel.setArg(index++, _states);
		_reconstructReceptiveKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_reconstructReceptiveKernel.setArg(index++, _receptiveReconstruction);
		_reconstructReceptiveKernel.setArg(index++, inputDims);
		_reconstructReceptiveKernel.setArg(index++, dims);
		_reconstructReceptiveKernel.setArg(index++, dimsToInputDims);
		_reconstructReceptiveKernel.setArg(index++, inputDimsToDims);
		_reconstructReceptiveKernel.setArg(index++, _receptiveRadius);
		_reconstructReceptiveKernel.setArg(index++, reverseReceptiveRadii);

		cs.getQueue().enqueueNDRangeKernel(_reconstructReceptiveKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
	}

	// Reconstruction - recurrent
	{
		int index = 0;

		_reconstructRecurrentKernel.setArg(index++, _states);
		_reconstructRecurrentKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_reconstructRecurrentKernel.setArg(index++, _recurrentReconstruction);
		_reconstructRecurrentKernel.setArg(index++, dims);
		_reconstructRecurrentKernel.setArg(index++, _recurrentRadius);

		cs.getQueue().enqueueNDRangeKernel(_reconstructRecurrentKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Error - receptive
	{
		int index = 0;

		_errorKernel.setArg(index++, inputs);
		_errorKernel.setArg(index++, _receptiveReconstruction);
		_errorKernel.setArg(index++, _receptiveErrors);
		
		cs.getQueue().enqueueNDRangeKernel(_errorKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
	}

	// Error - recurrent
	{
		int index = 0;

		_errorKernel.setArg(index++, _states);
		_errorKernel.setArg(index++, _recurrentReconstruction);
		_errorKernel.setArg(index++, _recurrentErrors);

		cs.getQueue().enqueueNDRangeKernel(_errorKernel, cl::NullRange, cl::NDRange(_width, _height));
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

		_learnKernel.setArg(index++, _receptiveErrors);
		_learnKernel.setArg(index++, _recurrentErrors);
		_learnKernel.setArg(index++, _activations);
		_learnKernel.setArg(index++, _inhibitions);
		_learnKernel.setArg(index++, _states);
		_learnKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_learnKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_learnKernel.setArg(index++, _hiddenHiddenWeightsPrev);
		_learnKernel.setArg(index++, _biasesPrev);
		_learnKernel.setArg(index++, _hiddenVisibleWeights);
		_learnKernel.setArg(index++, _hiddenHiddenPrevWeights);
		_learnKernel.setArg(index++, _hiddenHiddenWeights);
		_learnKernel.setArg(index++, _biases);
		_learnKernel.setArg(index++, inputDims);
		_learnKernel.setArg(index++, dims);
		_learnKernel.setArg(index++, dimsToInputDims);
		_learnKernel.setArg(index++, _receptiveRadius);
		_learnKernel.setArg(index++, _recurrentRadius);
		_learnKernel.setArg(index++, _inhibitionRadius);
		_learnKernel.setArg(index++, learningRates);
		_learnKernel.setArg(index++, sparsity);
		_learnKernel.setArg(index++, sparsity * sparsity);

		cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_width, _height));
	}
}

void RecurrentSparseCoder2D::stepEnd() {
	std::swap(_states, _statesPrev);
	std::swap(_hiddenVisibleWeights, _hiddenVisibleWeightsPrev);
	std::swap(_hiddenHiddenPrevWeights, _hiddenHiddenPrevWeightsPrev);
	std::swap(_hiddenHiddenWeights, _hiddenHiddenWeightsPrev);
	std::swap(_biases, _biasesPrev);
}