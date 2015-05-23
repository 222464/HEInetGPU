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

	_receptiveReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_receptiveError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_recurrentError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

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

void RecurrentSparseCoder2D::update(sys::ComputeSystem &cs, const cl::Image2D &input) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth - 1) / static_cast<float>(_width - 1), static_cast<float>(_inputHeight - 1) / static_cast<float>(_height - 1) };

	// Activate
	{
		int index = 0;

		_activateKernel.setArg(index++, input);
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

		

		cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Reconstruction - receptive
	{
		int index = 0;



		cs.getQueue().enqueueNDRangeKernel(_reconstructReceptiveKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
	}

	// Reconstruction - recurrent
	{
		int index = 0;



		cs.getQueue().enqueueNDRangeKernel(_reconstructRecurrentKernel, cl::NullRange, cl::NDRange(_width, _height));
	}
}

void RecurrentSparseCoder2D::learn(sys::ComputeSystem &cs, const cl::Image2D &input, float alpha, float beta, float gamma, float delta) {

}

void RecurrentSparseCoder2D::stepEnd() {
	std::swap(_states, _statesPrev);
	std::swap(_hiddenVisibleWeights, _hiddenVisibleWeightsPrev);
	std::swap(_hiddenHiddenPrevWeights, _hiddenHiddenPrevWeightsPrev);
	std::swap(_hiddenHiddenWeights, _hiddenHiddenWeightsPrev);
	std::swap(_biases, _biasesPrev);
}