#pragma once

#include "RecurrentSparseCoder2D.h"

namespace htsl {
	class HTSL {
	public:
		// Kernels this system uses
		struct Kernels {
			cl::Kernel _predictionInitializeKernel;
	
			cl::Kernel _predictKernel;
			cl::Kernel _predictionLearnKernel;

			// Load kernels from program
			void loadFromProgram(sys::ComputeProgram &program);
		};
	private:
		std::vector<RecurrentSparseCoder2D> _rscLayers;

		int _predictionRadiusFromE;
		int _predictionRadiusFromI;

		cl::Image2D _prediction;
		RecurrentSparseCoder2D::Weights2D _predictionFromEWeights;
		RecurrentSparseCoder2D::Weights2D _predictionFromIWeights;

		std::shared_ptr<Kernels> _kernels;

	public:
		// Randomly initialized weights
		void createRandom(const std::vector<RecurrentSparseCoder2D::Configuration> &rscConfigs,
			int predictionRadiusFromE, int predictionRadiusFromI,
			float minInitEWeight, float maxInitEWeight,
			float minInitIWeight, float maxInitIWeight,
			float initEThreshold, float initIThreshold,
			sys::ComputeSystem &cs, const std::shared_ptr<RecurrentSparseCoder2D::Kernels> &rscKernels,
			const std::shared_ptr<Kernels> &htslKernels, std::mt19937 &generator);

		// Run through a simulation step
		void update(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage, float eta, float homeoDecay);

		// Get prediction
		void predict(sys::ComputeSystem &cs);

		// Learn (seperate from simulation step)
		void learn(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage,
			float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
			float sparsity);

		// Learn prediction
		void learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha);

		// End step (buffer swap)
		void stepEnd();

		const std::vector<RecurrentSparseCoder2D> &getRSCLayers() const {
			return _rscLayers;
		}
	};

	void generateConfigsFromSizes(cl_int2 inputSize, const std::vector<cl_int2> &layerESizes, const std::vector<cl_int2> &layerISizes, std::vector<RecurrentSparseCoder2D::Configuration> &configs);
}