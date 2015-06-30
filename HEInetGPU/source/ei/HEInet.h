#pragma once

#include "EIlayer.h"

namespace ei {
	class HEInet {
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
		std::vector<EIlayer> _eiLayers;

		int _predictionRadiusFromE;
		int _predictionRadiusFromI;

		std::shared_ptr<Kernels> _kernels;

	public:
		cl::Image2D _prediction;
		cl::Image2D _predictionPrev;
		cl::Image2D _eShortAveragePrevIter;
		cl::Image2D _iShortAveragePrevIter;

		EIlayer::Weights2D _predictionFromEWeights;
		EIlayer::Weights2D _predictionFromIWeights;

		// Randomly initialized weights
		void createRandom(const std::vector<EIlayer::Configuration> &eilConfigs,
			int predictionRadiusFromE, int predictionRadiusFromI,
			float minInitEWeight, float maxInitEWeight,
			float minInitIWeight, float maxInitIWeight,
			float initEThreshold, float initIThreshold,
			float sparsityE, float sparsityI,
			sys::ComputeSystem &cs, const std::shared_ptr<EIlayer::Kernels> &eilKernels,
			const std::shared_ptr<Kernels> &heiKernels, std::mt19937 &generator);

		// Run through an example step (multiple simulation steps)
		void update(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage, int iter, float eta, float longAverageDecay);

		// Get prediction
		void predict(sys::ComputeSystem &cs);

		// Learn
		void learn(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage,
			float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
			float sparsityE, float sparsityI);

		// Learn prediction
		void learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha);

		// End example step (buffer swap)
		void exStepEnd(sys::ComputeSystem &cs);

		const std::vector<EIlayer> &getEIlayers() const {
			return _eiLayers;
		}

		int getPredictionRadiusFromE() const {
			return _predictionRadiusFromE;
		}

		int getPredictionRadiusFromI() const {
			return _predictionRadiusFromI;
		}
	};

	void generateConfigsFromSizes(cl_int2 inputSize, const std::vector<cl_int2> &layerESizes, const std::vector<cl_int2> &layerISizes, std::vector<EIlayer::Configuration> &configs);
}