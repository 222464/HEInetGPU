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

// Sampler definitions
constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_NONE |
	CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_NONE |
	CLK_FILTER_NEAREST;

// RNG
float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------- RSC ---------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------

// Random weight initialization - excitatory
void kernel rsc_eInitialize(write_only image3d_t eFeedForwardWeights,
	write_only image3d_t eFeedBackWeights,
	int eFeedForwardSize, int eFeedBackSize,
	float minInitEWeight, float maxInitEWeight,
	float minInitIWeight, float maxInitIWeight,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < eFeedForwardSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitEWeight - minInitEWeight) + minInitEWeight;

		write_imagef(eFeedForwardWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < eFeedBackSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitIWeight - minInitIWeight) + minInitIWeight;

		write_imagef(eFeedBackWeights, weightPosition, (float4)(weight));
	}
}

// Random weight initialization - inhibitory
void kernel rsc_iInitialize(write_only image3d_t iFeedForwardWeights,
	write_only image3d_t iFeedBackWeights,
	write_only image3d_t iLateralWeights,
	int iFeedForwardSize, int iLateralSize, int iFeedBackSize,
	float minInitEWeight, float maxInitEWeight,
	float minInitIWeight, float maxInitIWeight,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < iFeedForwardSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitEWeight - minInitEWeight) + minInitEWeight;

		write_imagef(iFeedForwardWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < iLateralSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitIWeight - minInitIWeight) + minInitIWeight;

		write_imagef(iLateralWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < iFeedBackSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitIWeight - minInitIWeight) + minInitIWeight;

		write_imagef(iFeedBackWeights, weightPosition, (float4)(weight));
	}
}

void kernel rsc_eActivate(read_only image2d_t feedForwardInput, read_only image2d_t iStatesPrev,
	read_only image3d_t eFeedForwardWeightsPrev, read_only image3d_t eFeedBackWeightsPrev, read_only image2d_t eThresholdsPrev,
	read_only image2d_t eActivationsPrev, read_only image2d_t eStatesPrev,
	write_only image2d_t eActivations, write_only image2d_t eStates,
	int2 eFeedForwardDims, int2 eDims, int2 iDims,
	float2 eDimsToEFeedForwardDims, float2 eDimsToIDims,
	int eFeedForwardRadius, int eFeedBackRadius,
	float eta, float homeoDecay)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 feedForwardCenterPosition = (int2)(position.x * eDimsToEFeedForwardDims.x + 0.5f, position.y * eDimsToEFeedForwardDims.y + 0.5f);
	int2 feedBackCenterPosition = (int2)(position.x * eDimsToIDims.x + 0.5f, position.y * eDimsToIDims.y + 0.5f);

	int wi = 0;

	float excitation = 0.0f;
	float inhibition = 0.0f;

	// Feed forward (excitatory)
	for (int dx = -eFeedForwardRadius; dx <= eFeedForwardRadius; dx++)
		for (int dy = -eFeedForwardRadius; dy <= eFeedForwardRadius; dy++) {
			int2 feedForwardPosition = (int2)(feedForwardCenterPosition.x + dx, feedForwardCenterPosition.y + dy);

			if (feedForwardPosition.x >= 0 && feedForwardPosition.x < eFeedForwardDims.x && feedForwardPosition.y >= 0 && feedForwardPosition.y < eFeedForwardDims.y) {
				float input = read_imagef(feedForwardInput, defaultUnnormalizedSampler, feedForwardPosition).x;

				float weight = read_imagef(eFeedForwardWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				excitation += input * weight;
			}

			wi++;
		}

	wi = 0;

	// Feed back (inhibitory)
	for (int dx = -eFeedBackRadius; dx <= eFeedBackRadius; dx++)
		for (int dy = -eFeedBackRadius; dy <= eFeedBackRadius; dy++) {
			int2 feedBackPosition = (int2)(feedBackCenterPosition.x + dx, feedBackCenterPosition.y + dy);

			if (feedBackPosition.x >= 0 && feedBackPosition.x < iDims.x && feedBackPosition.y >= 0 && feedBackPosition.y < iDims.y) {
				float input = read_imagef(iStatesPrev, defaultUnnormalizedSampler, feedBackPosition).x;

				float weight = read_imagef(eFeedBackWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				inhibition += input * weight;
			}

			wi++;
		}

	float activationPrev = read_imagef(eActivationsPrev, defaultUnnormalizedSampler, position).x;

	float activation = (1.0f - eta) * activationPrev + (excitation - inhibition);

	float thresholdPrev = read_imagef(eThresholdsPrev, defaultUnnormalizedSampler, position).x;

	float2 statePrev = read_imagef(eStatesPrev, defaultUnnormalizedSampler, position).xy;

	float state = 0.0f;

	if (activation > thresholdPrev) {
		state = 1.0f;

		activation = 0.0f;
	}

	write_imagef(eActivations, position, (float4)(activation));
	write_imagef(eStates, position, (float4)(state, (1.0f - homeoDecay) * statePrev.y + homeoDecay * state, 0.0f, 0.0f));
}

void kernel rsc_iActivate(read_only image2d_t eStates, read_only image2d_t feedBackInput,
	read_only image3d_t iFeedForwardWeightsPrev, read_only image3d_t iLateralWeightsPrev, read_only image3d_t iFeedBackWeightsPrev, read_only image2d_t iThresholdsPrev,
	read_only image2d_t iActivationsPrev, read_only image2d_t iStatesPrev,
	write_only image2d_t iActivations, write_only image2d_t iStates,
	int2 eDims, int2 iDims, int2 iFeedBackDims,
	float2 iDimsToEDims, float2 iDimsToFeedBackDims,
	int iFeedForwardRadius, int iLateralRadius, int iFeedBackRadius,
	float eta, float homeoDecay)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 feedForwardCenterPosition = (int2)(position.x * iDimsToEDims.x + 0.5f, position.y * iDimsToEDims.y + 0.5f);
	int2 feedBackCenterPosition = (int2)(position.x * iDimsToFeedBackDims.x + 0.5f, position.y * iDimsToFeedBackDims.y + 0.5f);

	int wi = 0;

	float excitation = 0.0f;
	float inhibition = 0.0f;

	// Feed forward (excitatory)
	for (int dx = -iFeedForwardRadius; dx <= iFeedForwardRadius; dx++)
		for (int dy = -iFeedForwardRadius; dy <= iFeedForwardRadius; dy++) {
			int2 feedForwardPosition = (int2)(feedForwardCenterPosition.x + dx, feedForwardCenterPosition.y + dy);

			if (feedForwardPosition.x >= 0 && feedForwardPosition.x < eDims.x && feedForwardPosition.y >= 0 && feedForwardPosition.y < eDims.y) {
				float input = read_imagef(eStates, defaultUnnormalizedSampler, feedForwardPosition).x;

				float weight = read_imagef(iFeedForwardWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				excitation += input * weight;
			}

			wi++;
		}

	wi = 0;

	// Feed back (inhibitory)
	for (int dx = -iFeedBackRadius; dx <= iFeedBackRadius; dx++)
		for (int dy = -iFeedBackRadius; dy <= iFeedBackRadius; dy++) {
			int2 feedBackPosition = (int2)(feedBackCenterPosition.x + dx, feedBackCenterPosition.y + dy);

			if (feedBackPosition.x >= 0 && feedBackPosition.x < iFeedBackDims.x && feedBackPosition.y >= 0 && feedBackPosition.y < iFeedBackDims.y) {
				float input = read_imagef(feedBackInput, defaultUnnormalizedSampler, feedBackPosition).x;

				float weight = read_imagef(iFeedBackWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				inhibition += input * weight;
			}

			wi++;
		}

	wi = 0;

	// Lateral (inhibitory)
	for (int dx = -iLateralRadius; dx <= iLateralRadius; dx++)
		for (int dy = -iLateralRadius; dy <= iLateralRadius; dy++) {
			if (dx != 0 || dy != 0) {
				int2 lateralPosition = (int2)(position.x + dx, position.y + dy);

				if (lateralPosition.x >= 0 && lateralPosition.x < iDims.x && lateralPosition.y >= 0 && lateralPosition.y < iDims.y) {
					float input = read_imagef(iStatesPrev, defaultUnnormalizedSampler, lateralPosition).x;

					float weight = read_imagef(iLateralWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

					inhibition += input * weight;
				}
			}

			wi++;
		}

	float activationPrev = read_imagef(iActivationsPrev, defaultUnnormalizedSampler, position).x;

	float activation = (1.0f - eta) * activationPrev + (excitation - inhibition);

	float thresholdPrev = read_imagef(iThresholdsPrev, defaultUnnormalizedSampler, position).x;

	float2 statePrev = read_imagef(iStatesPrev, defaultUnnormalizedSampler, position).xy;

	float state = 0.0f;

	if (activation > thresholdPrev) {
		state = 1.0f;

		activation = 0.0f;
	}

	write_imagef(iActivations, position, (float4)(activation));
	write_imagef(iStates, position, (float4)(state, (1.0f - homeoDecay) * statePrev.y + homeoDecay * state, 0.0f, 0.0f));
}

// Learn - excitatory
void kernel rsc_eLearn(read_only image2d_t feedForwardInput, read_only image2d_t iStatesPrev, read_only image2d_t eStates,
	read_only image3d_t eFeedForwardWeightsPrev, read_only image3d_t eFeedBackWeightsPrev, read_only image2d_t eThresholdsPrev,
	write_only image3d_t eFeedForwardWeights, write_only image3d_t eFeedBackWeights, write_only image2d_t eThresholds,
	int2 eFeedForwardDims, int2 eDims, int2 iDims,
	float2 eDimsToEFeedForwardDims, float2 eDimsToIDims,
	int eFeedForwardRadius, int eFeedBackRadius,
	float alpha, float beta, float delta, float sparsity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 state = read_imagef(eStates, defaultUnnormalizedSampler, position).xy;

	int2 feedForwardCenterPosition = (int2)(position.x * eDimsToEFeedForwardDims.x + 0.5f, position.y * eDimsToEFeedForwardDims.y + 0.5f);
	int2 feedBackCenterPosition = (int2)(position.x * eDimsToIDims.x + 0.5f, position.y * eDimsToIDims.y + 0.5f);

	int wi = 0;

	// Feed forward (excitatory)
	for (int dx = -eFeedForwardRadius; dx <= eFeedForwardRadius; dx++)
		for (int dy = -eFeedForwardRadius; dy <= eFeedForwardRadius; dy++) {
			int2 feedForwardPosition = (int2)(feedForwardCenterPosition.x + dx, feedForwardCenterPosition.y + dy);

			if (feedForwardPosition.x >= 0 && feedForwardPosition.x < eFeedForwardDims.x && feedForwardPosition.y >= 0 && feedForwardPosition.y < eFeedForwardDims.y) {
				float input = read_imagef(feedForwardInput, defaultUnnormalizedSampler, feedForwardPosition).x;

				float weightPrev = read_imagef(eFeedForwardWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = weightPrev + alpha * state.x * (input - weightPrev);

				write_imagef(eFeedForwardWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Feed back (inhibitory)
	for (int dx = -eFeedBackRadius; dx <= eFeedBackRadius; dx++)
		for (int dy = -eFeedBackRadius; dy <= eFeedBackRadius; dy++) {
			int2 feedBackPosition = (int2)(feedBackCenterPosition.x + dx, feedBackCenterPosition.y + dy);

			if (feedBackPosition.x >= 0 && feedBackPosition.x < iDims.x && feedBackPosition.y >= 0 && feedBackPosition.y < iDims.y) {
				float2 input = read_imagef(iStatesPrev, defaultUnnormalizedSampler, feedBackPosition).xy;

				float weightPrev = read_imagef(eFeedBackWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = fmax(0.0f, weightPrev + beta * (state.x * input.x - state.y * input.y * (1.0f + weightPrev)));

				write_imagef(eFeedBackWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	float thresholdPrev = read_imagef(eThresholdsPrev, defaultUnnormalizedSampler, position).x;

	float threshold = thresholdPrev + delta * (state.x - sparsity);

	write_imagef(eThresholds, position, (float4)(threshold));
}

// Learn - inhibitory
void kernel rsc_iLearn(read_only image2d_t eStates, read_only image2d_t iStatesPrev, read_only image2d_t feedBackInputs, read_only image2d_t iStates,
	read_only image3d_t iFeedForwardWeightsPrev, read_only image3d_t iLateralWeightsPrev, read_only image3d_t iFeedBackWeightsPrev, read_only image2d_t iThresholdsPrev,
	write_only image3d_t iFeedForwardWeights, write_only image3d_t iLateralWeights, write_only image3d_t iFeedBackWeights, write_only image2d_t iThresholds,
	int2 eDims, int2 iDims, int2 iFeedBackDims,
	float2 iDimsToEDims, float2 iDimsToFeedBackDims,
	int iFeedForwardRadius, int iLateralRadius, int iFeedBackRadius,
	float alpha, float beta, float gamma, float delta, float sparsity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 feedForwardCenterPosition = (int2)(position.x * iDimsToEDims.x + 0.5f, position.y * iDimsToEDims.y + 0.5f);
	int2 feedBackCenterPosition = (int2)(position.x * iDimsToFeedBackDims.x + 0.5f, position.y * iDimsToFeedBackDims.y + 0.5f);

	float2 state = read_imagef(iStates, defaultUnnormalizedSampler, position).xy;

	int wi = 0;

	// Feed forward (excitatory)
	for (int dx = -iFeedForwardRadius; dx <= iFeedForwardRadius; dx++)
		for (int dy = -iFeedForwardRadius; dy <= iFeedForwardRadius; dy++) {
			int2 feedForwardPosition = (int2)(feedForwardCenterPosition.x + dx, feedForwardCenterPosition.y + dy);

			if (feedForwardPosition.x >= 0 && feedForwardPosition.x < eDims.x && feedForwardPosition.y >= 0 && feedForwardPosition.y < eDims.y) {
				float2 input = read_imagef(eStates, defaultUnnormalizedSampler, feedForwardPosition).xy;

				float weightPrev = read_imagef(iFeedForwardWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = fmax(0.0f, weightPrev + alpha * (state.x * input.x - state.y * input.y * (1.0f + weightPrev)));

				write_imagef(iFeedForwardWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Feed back (inhibitory)
	for (int dx = -iFeedBackRadius; dx <= iFeedBackRadius; dx++)
		for (int dy = -iFeedBackRadius; dy <= iFeedBackRadius; dy++) {
			int2 feedBackPosition = (int2)(feedBackCenterPosition.x + dx, feedBackCenterPosition.y + dy);

			if (feedBackPosition.x >= 0 && feedBackPosition.x < iFeedBackDims.x && feedBackPosition.y >= 0 && feedBackPosition.y < iFeedBackDims.y) {
				float2 input = read_imagef(feedBackInputs, defaultUnnormalizedSampler, feedBackPosition).xy;

				float weightPrev = read_imagef(iFeedBackWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				//float weight = weightPrev + beta * state.x * (input - weightPrev);

				float weight = fmax(0.0f, weightPrev + alpha * (state.x * input.x - state.y * input.y * (1.0f + weightPrev)));

				write_imagef(iFeedBackWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Lateral (inhibitory)
	for (int dx = -iLateralRadius; dx <= iLateralRadius; dx++)
		for (int dy = -iLateralRadius; dy <= iLateralRadius; dy++) {
			int2 lateralPosition = (int2)(position.x + dx, position.y + dy);

			if (lateralPosition.x >= 0 && lateralPosition.x < iDims.x && lateralPosition.y >= 0 && lateralPosition.y < iDims.y) {
				float2 input = read_imagef(iStatesPrev, defaultUnnormalizedSampler, lateralPosition).xy;

				float weightPrev = read_imagef(iLateralWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = fmax(0.0f, weightPrev + gamma * (state.x * input.x - state.y * input.y * (1.0f + weightPrev)));

				write_imagef(iLateralWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	float thresholdPrev = read_imagef(iThresholdsPrev, defaultUnnormalizedSampler, position).x;

	float threshold = thresholdPrev + delta * (state.x - sparsity);

	write_imagef(iThresholds, position, (float4)(threshold));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------- HTSL --------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------

// Initialize prediction weights
void kernel htsl_predictionInitialize(write_only image3d_t predictionWeightsFromE, write_only image3d_t predictionWeightsFromI,
	int predictionFromESize, int predictionFromISize,
	float minInitWeight, float maxInitWeight, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < predictionFromESize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(predictionWeightsFromE, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < predictionFromISize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(predictionWeightsFromI, weightPosition, (float4)(weight));
	}
}

// Perceptron from eStates and iStates that forms predictions
void kernel htsl_predict(read_only image2d_t eStates, read_only image2d_t iStates,
	read_only image3d_t predictionFromEWeightsPrev, read_only image3d_t predictionFromIWeightsPrev,
	write_only image2d_t predictions,
	float2 eFeedForwardDimsToEDims, float2 eFeedForwardDimsToIDims,
	int2 eDims, int2 iDims,
	int predictionRadiusFromE, int predictionRadiusFromI)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 eCenterPosition = (int2)(position.x * eFeedForwardDimsToEDims.x + 0.5f, position.y * eFeedForwardDimsToEDims.y + 0.5f);
	int2 iCenterPosition = (int2)(position.x * eFeedForwardDimsToIDims.x + 0.5f, position.y * eFeedForwardDimsToIDims.y + 0.5f);

	int wi = 0;

	float sum = 0.0f;

	for (int dx = -predictionRadiusFromE; dx <= predictionRadiusFromE; dx++)
		for (int dy = -predictionRadiusFromE; dy <= predictionRadiusFromE; dy++) {
			int2 ePosition = (int2)(eCenterPosition.x + dx, eCenterPosition.y + dy);

			if (ePosition.x >= 0 && ePosition.x < eDims.x && ePosition.y >= 0 && ePosition.y < eDims.y) {
				float input = read_imagef(eStates, defaultUnnormalizedSampler, ePosition).x;

				float weight = read_imagef(predictionFromEWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				sum += weight * input;
			}

			wi++;
		}

	wi = 0;

	for (int dx = -predictionRadiusFromI; dx <= predictionRadiusFromI; dx++)
		for (int dy = -predictionRadiusFromI; dy <= predictionRadiusFromI; dy++) {
			int2 iPosition = (int2)(iCenterPosition.x + dx, iCenterPosition.y + dy);

			if (iPosition.x >= 0 && iPosition.x < iDims.x && iPosition.y >= 0 && iPosition.y < iDims.y) {
				float input = read_imagef(iStates, defaultUnnormalizedSampler, iPosition).x;

				float weight = read_imagef(predictionFromIWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				sum += weight * input;
			}

			wi++;
		}

	write_imagef(predictions, position, (float4)(sum));
}

// Learn perceptron
void kernel htsl_predictionLearn(read_only image2d_t eStates, read_only image2d_t iStates,
	read_only image2d_t feedForwardInput, read_only image2d_t predictions,
	read_only image3d_t predictionFromEWeightsPrev, read_only image3d_t predictionFromIWeightsPrev,
	write_only image3d_t predictionFromEWeights, write_only image3d_t predictionFromIWeights,
	float2 eFeedForwardDimsToEDims, float2 eFeedForwardDimsToIDims,
	int2 eDims, int2 iDims,
	int predictionRadiusFromE, int predictionRadiusFromI,
	float alpha)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 eCenterPosition = (int2)(position.x * eFeedForwardDimsToEDims.x + 0.5f, position.y * eFeedForwardDimsToEDims.y + 0.5f);
	int2 iCenterPosition = (int2)(position.x * eFeedForwardDimsToIDims.x + 0.5f, position.y * eFeedForwardDimsToIDims.y + 0.5f);

	float target = read_imagef(feedForwardInput, defaultUnnormalizedSampler, position).x;
	float prediction = read_imagef(predictions, defaultUnnormalizedSampler, position).x;

	float alphaError = alpha * (target - prediction);

	int wi = 0;

	for (int dx = -predictionRadiusFromE; dx <= predictionRadiusFromE; dx++)
		for (int dy = -predictionRadiusFromE; dy <= predictionRadiusFromE; dy++) {
			int2 ePosition = (int2)(eCenterPosition.x + dx, eCenterPosition.y + dy);

			if (ePosition.x >= 0 && ePosition.x < eDims.x && ePosition.y >= 0 && ePosition.y < eDims.y) {
				float input = read_imagef(eStates, defaultUnnormalizedSampler, ePosition).x;

				float weightPrev = read_imagef(predictionFromEWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = weightPrev + alphaError * input;

				write_imagef(predictionFromEWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	for (int dx = -predictionRadiusFromI; dx <= predictionRadiusFromI; dx++)
		for (int dy = -predictionRadiusFromI; dy <= predictionRadiusFromI; dy++) {
			int2 iPosition = (int2)(iCenterPosition.x + dx, iCenterPosition.y + dy);

			if (iPosition.x >= 0 && iPosition.x < iDims.x && iPosition.y >= 0 && iPosition.y < iDims.y) {
				float input = read_imagef(iStates, defaultUnnormalizedSampler, iPosition).x;

				float weightPrev = read_imagef(predictionFromIWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float weight = weightPrev + alphaError * input;

				write_imagef(predictionFromIWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}
}

// Sum spikes - excitatory
void kernel htsl_sumSpikesE(read_only image2d_t eStates,
	read_only image2d_t sumSpikesEPrev,
	write_only image2d_t sumSpikesE,
	float scalar)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float ssEPrev = read_imagef(sumSpikesEPrev, position).x;

	float eState = read_imagef(eStates, position).x;

	float ssE = ssEPrev + eState * scalar;

	write_imagef(sumSpikesE, position, (float4)(ssE));
}

// Sum spikes - inhibitory
void kernel htsl_sumSpikesI(read_only image2d_t iStates,
	read_only image2d_t sumSpikesIPrev,
	write_only image2d_t sumSpikesI,
	float scalar)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float ssIPrev = read_imagef(sumSpikesIPrev, position).x;

	float iState = read_imagef(iStates, position).x;

	float ssI = ssIPrev + iState * scalar;

	write_imagef(sumSpikesI, position, (float4)(ssI));
}